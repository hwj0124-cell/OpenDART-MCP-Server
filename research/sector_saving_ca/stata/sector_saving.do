* 가계·기업 저축 급변 이후 부문별 저축-투자 차(S-I)의 지속성
* OECD 제도부문 계정 패널, 이벤트 스터디형 국소투영
* 필요 패키지: ssc install ftools / ssc install reghdfe
* 이 파일이 있는 폴더를 작업 폴더로 지정하고 실행 (cd "...")

version 15
clear all
set more off
local reps 499                  // 부트스트랩 반복 수 (빠르게 볼 때는 50 정도로)

import delimited "sector_saving_panel.csv", clear
encode country, gen(cid)
xtset cid year

gen div_net = div_paid - div_recv_corp       // 순배당 지급
gen reinv   = 100 * gcf_corp / prof          // 재투자율(총자본형성/영업잉여)
gen payout  = 100 * div_net / prof           // 순배당 성향

label var nl_hh   "가계 순대출(S-I), GDP 대비 %"
label var nl_corp "비금융법인 순대출(S-I), GDP 대비 %"
label var hh_rate "가계 총저축률, 가처분소득 대비 %"
label var prof    "비금융법인 총영업잉여, GDP 대비 %"
label var tot_na_g "교역조건 변화율(수출/수입 디플레이터), %"


*** 1. 사례 선정 ***

* 국가별 분위수. pandas 기본값(선형보간)과 맞춤
capture program drop qtile
program qtile
    args v p out
    tempvar n k
    bys cid (`v'): egen `n' = count(`v')
    gen double `k' = (`n' - 1) * `p'
    bys cid (`v'): gen double `out' = `v'[floor(`k') + 1] + ///
        (`k' - floor(`k')) * (`v'[min(floor(`k') + 2, `n')] - `v'[floor(`k') + 1])
end

gen dhh   = D.hh_rate
gen dprof = D.prof
gen tot   = tot_na_g
foreach v in dhh tot {
    qtile `v' .1 `v'_lo
    qtile `v' .9 `v'_hi
}
sort cid year

* 후보: 국가별 상하위 10%, 시작 연도 성장률 >= 0, 2020-22년 제외
* 교역조건은 영업잉여가 같은 방향으로 움직인 해만
gen byte ok = gdp_g >= 0 & !missing(gdp_g) & !inrange(year, 2020, 2022)
gen byte c_hh   = ok & !missing(dhh) & (dhh >= dhh_hi | dhh <= dhh_lo)
gen byte c_corp = ok & !missing(tot) & (tot >= tot_hi | tot <= tot_lo) & sign(dprof) == sign(tot)
gen s_hh   = sign(dhh)
gen s_corp = sign(tot)

* 같은 국가에서 직전 사례 후 4년 이내는 제외 (앞에서부터 순서대로)
foreach s in hh corp {
    gen byte e_`s' = 0
    local last -99
    forvalues i = 1/`=_N' {
        if cid[`i'] != cid[`i' - 1] local last -99
        if c_`s'[`i'] & year[`i'] - `last' > 4 {
            qui replace e_`s' = 1 in `i'
            local last = year[`i']
        }
    }
}

* 시작 연도 자기 부문 S-I가 충격 방향으로 움직인 사례만.
* 가계는 소비가 주도한 경우(소비/GDP 변화가 소득/GDP 변화보다 큼)만
gen z_hh = cond(e_hh & sign(D.nl_hh) == s_hh & !missing(D.cons_hh, D.yd_hh) ///
    & -s_hh * D.cons_hh >= s_hh * D.yd_hh, s_hh, 0)
gen z_corp = cond(e_corp & sign(D.nl_corp) == s_corp, s_corp, 0)

count if z_hh != 0
local n_hh = r(N)
count if z_corp != 0
local n_corp = r(N)
list country year z_hh   if z_hh   != 0, noobs sep(0)
list country year z_corp if z_corp != 0, noobs sep(0)


*** 2. 국소투영 변수 ***

local n_hh_v   nl_hh yd_hh cons_hh gcf_hh        // S-I 시작 반응으로 정규화
local n_corp_v nl_corp prof div_net gcf_corp
local r_hh_v   hh_rate                           // %p 그대로
local r_corp_v reinv payout

foreach v in `n_hh_v' `n_corp_v' `r_hh_v' `r_corp_v' {
    gen `v'_0 = `v' - L.`v'
    forvalues h = 1/6 {
        gen `v'_`h' = F`h'.`v' - L.`v'
    }
}

* 사건 전 자기 부문 S-I 변화 두 시차를 통제
* 가중치: 시작 연도 S-I 식(무가중) 잔차의 국가별 분산의 역수
foreach s in hh corp {
    gen dl1_`s' = L.D.nl_`s'
    gen dl2_`s' = L2.D.nl_`s'
    qui reghdfe nl_`s'_0 z_`s' dl1_`s' dl2_`s', absorb(cid year) keepsingletons residuals(res_`s')
    bys cid: egen sd_`s' = sd(res_`s')
    gen w_`s' = 1 / sd_`s'^2
}
sort cid year
save "sector_saving.dta", replace


*** 3. 추정 ***

* y(t+h) - y(t-1) = g_h * z + 두 시차 + 국가FE + 연도FE, 가중최소제곱
* vars()는 g_h / g_0(S-I), raw()는 g_h 그대로
capture program drop lproj
program lproj, rclass
    syntax, s(string) [vars(string) raw(string) fe(name)]
    if "`fe'" == "" local fe cid
    local x z_`s' dl1_`s' dl2_`s'
    tempname g0
    qui reghdfe nl_`s'_0 `x' [aw = w_`s'], absorb(`fe' year) keepsingletons
    scalar `g0' = _b[z_`s']
    foreach v in `vars' `raw' {
        local d = cond(strpos(" `raw' ", " `v' "), 1, `g0')
        forvalues h = 0/6 {
            qui reghdfe `v'_`h' `x' [aw = w_`s'], absorb(`fe' year) keepsingletons
            return scalar `v'_`h' = _b[z_`s'] / `d'
        }
    }
end

tempfile res bs
tempname pf
postfile `pf' str4 s str10 v byte h double(b lo hi) using `res'
foreach s in hh corp {
    * 점추정: 그림에 쓰는 변수 전부
    lproj, s(`s') vars(`n_`s'_v') raw(`r_`s'_v')
    foreach v in `n_`s'_v' `r_`s'_v' {
        forvalues h = 0/6 {
            local b_`v'_`h' = r(`v'_`h')
        }
    }

    * 구간: 그림 1(S-I)과 그림 4(성향)만. 국가 단위 복원추출, 가중치는 고정
    local exp
    foreach v in nl_`s' `r_`s'_v' {
        forvalues h = 0/6 {
            local exp `exp' `v'_`h'=r(`v'_`h')
        }
    }
    capture drop bid
    gen bid = cid
    bootstrap `exp', reps(`reps') seed(1) cluster(cid) idcluster(bid) saving(`bs', replace) nodots: ///
        lproj, s(`s') vars(nl_`s') raw(`r_`s'_v') fe(bid)

    preserve
    use `bs', clear
    foreach v in `n_`s'_v' `r_`s'_v' {
        forvalues h = 0/6 {
            local lo = .
            local hi = .
            capture confirm variable `v'_`h'
            if !_rc {
                _pctile `v'_`h', p(16 84)
                local lo = r(r1)
                local hi = r(r2)
            }
            post `pf' ("`s'") ("`v'") (`h') (`b_`v'_`h'') (`lo') (`hi')
        }
    }
    restore
}
postclose `pf'

use `res', clear
list s v h b lo hi if inlist(v, "nl_hh", "nl_corp"), sepby(v) noobs


*** 4. 그림 ***

graph set window fontface "Times New Roman"
local opt yline(0, lc(gs8) lw(thin)) xlabel(0/6) xtitle("Years after episode onset (h)") graphregion(color(white)) plotregion(margin(small))

* 그림 1: S-I 지속성
twoway (rarea lo hi h if v == "nl_hh", color(navy%15) lw(none)) ///
       (connected b h if v == "nl_hh", lc(navy) mc(navy) ms(O) msize(small)) ///
       (rarea lo hi h if v == "nl_corp", color(dkorange%15) lw(none)) ///
       (connected b h if v == "nl_corp", lc(dkorange) mc(dkorange) lp(dash) ms(O) msize(small)), ///
       `opt' ytitle("Saving-investment balance (impact = 1)") ///
       legend(order(2 "Households: consumption-driven saving shifts (N = `n_hh')" ///
                    4 "Corporations: terms-of-trade-driven profit shifts (N = `n_corp')") ///
              pos(7) ring(0) col(1) region(lc(none)) size(small))
graph export "fig1_si_persistence.png", replace width(2000)

* 그림 2·3: S-I 분해 (막대 합 = S-I). 같은 세로축
local corp_c bprof -bdiv_net -bgcf_corp
local corp_l `""Gross operating surplus" "Net dividends paid (-)" "Gross capital formation (-)""'
local corp_k navy "122 106 168" "181 84 28"
local hh_c -bcons_hh byd_hh -bgcf_hh
local hh_l `""Final consumption (-)" "Gross disposable income" "Gross capital formation (-)""'
local hh_k "122 106 168" navy "181 84 28"
local i 2
foreach s in corp hh {
    preserve
    keep if s == "`s'"
    keep v h b
    reshape wide b, i(h) j(v) string
    gen up0 = 0
    gen dn0 = 0
    forvalues k = 1/4 {
        if `k' < 4 gen c`k' = `: word `k' of ``s'_c''
        else gen c4 = bnl_`s' - c1 - c2 - c3
        gen lo`k' = cond(c`k' >= 0, up`=`k'-1', dn`=`k'-1')
        gen hi`k' = lo`k' + c`k'
        gen up`k' = up`=`k'-1' + max(c`k', 0)
        gen dn`k' = dn`=`k'-1' + min(c`k', 0)
    }
    local bars
    forvalues k = 1/3 {
        local bars `bars' (rbar lo`k' hi`k' h, barw(.6) color("`: word `k' of ``s'_k''") lw(none))
    }
    local lab = cond("`s'" == "corp", "corporate", "household")
    twoway `bars' (rbar lo4 hi4 h, barw(.6) color(gs11) lw(none)) ///
           (connected bnl_`s' h, lc(black) mc(white) mlc(black) ms(O)), ///
           `opt' ylabel(-.6(.2)1.2) ytitle("Contribution (impact S-I = 1)") ///
           title("Decomposition of the `lab' saving-investment balance", size(medium) pos(11)) ///
           legend(order(5 "S-I balance" 1 `"`: word 1 of ``s'_l''"' 2 `"`: word 2 of ``s'_l''"' ///
                        3 `"`: word 3 of ``s'_l''"' 4 "Other") rows(2) size(small) region(lc(none)))
    graph export "fig`i'_`s'_decomposition.png", replace width(2000)
    restore
    local ++i
}

* 그림 4: 가계 저축률 vs 기업 재투자율·배당성향 (%p, 정규화 안 함)
twoway (rarea lo hi h if v == "hh_rate", color(navy%15) lw(none)) ///
       (connected b h if v == "hh_rate", lc(navy) mc(navy) ms(O) msize(small)), ///
       `opt' ytitle("pp of disposable income") legend(off) name(a, replace) ///
       title("(a) Household saving rate, household episodes", size(medsmall) pos(11))
twoway (rarea lo hi h if v == "reinv", color(dkorange%15) lw(none)) ///
       (connected b h if v == "reinv", lc(dkorange) mc(dkorange) ms(O) msize(small)) ///
       (rarea lo hi h if v == "payout", color("122 106 168%15") lw(none)) ///
       (connected b h if v == "payout", lc("122 106 168") mc("122 106 168") lp(dash) ms(O) msize(small)), ///
       `opt' ytitle("pp of operating surplus") name(b, replace) ///
       title("(b) Corporations, terms-of-trade episodes", size(medsmall) pos(11)) ///
       legend(order(2 "Reinvestment ratio" 4 "Net payout ratio") pos(1) ring(0) col(1) region(lc(none)) size(small))
graph combine a b, rows(1) xsize(10) ysize(4.2) graphregion(color(white))
graph export "fig4_propensities.png", replace width(2400)
