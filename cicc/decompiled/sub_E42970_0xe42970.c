// Function: sub_E42970
// Address: 0xe42970
//
char __fastcall sub_E42970(_QWORD *a1, _DWORD *a2)
{
  unsigned __int64 v2; // rdx
  unsigned __int64 v4; // rdi
  int v5; // eax
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v8; // edx
  int v9; // edx
  int v10; // eax
  unsigned int v11; // r13d
  int v12; // eax
  __int64 i; // rax
  unsigned int v14; // edx
  __int64 v15; // rax
  int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  int v20; // edx
  int v21; // eax

  v2 = (unsigned __int64)(a1 + 731);
  v4 = (unsigned __int64)(a1 + 1);
  *(_QWORD *)(v4 + 5824) = 0;
  memset((void *)(v4 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * (((unsigned int)a1 - (v4 & 0xFFFFFFF8) + 5840) >> 3));
  *a1 = "__ashlhi3";
  a1[1] = "__ashlsi3";
  a1[2] = "__ashldi3";
  a1[3] = "__ashlti3";
  a1[4] = "__lshrhi3";
  a1[5] = "__lshrsi3";
  a1[6] = "__lshrdi3";
  a1[7] = "__lshrti3";
  a1[8] = "__ashrhi3";
  a1[9] = "__ashrsi3";
  a1[10] = "__ashrdi3";
  a1[11] = "__ashrti3";
  a1[12] = "__mulqi3";
  a1[13] = "__mulhi3";
  a1[14] = "__mulsi3";
  a1[15] = "__muldi3";
  a1[16] = "__multi3";
  a1[17] = "__mulosi4";
  a1[18] = "__mulodi4";
  a1[19] = "__muloti4";
  a1[20] = "__divqi3";
  a1[21] = "__divhi3";
  a1[22] = "__divsi3";
  a1[23] = "__divdi3";
  a1[24] = "__divti3";
  a1[25] = "__udivqi3";
  a1[26] = "__udivhi3";
  a1[27] = "__udivsi3";
  a1[28] = "__udivdi3";
  a1[29] = "__udivti3";
  a1[30] = "__modqi3";
  a1[31] = "__modhi3";
  a1[32] = "__modsi3";
  a1[33] = "__moddi3";
  a1[34] = "__modti3";
  a1[35] = "__umodqi3";
  a1[36] = "__umodhi3";
  a1[37] = "__umodsi3";
  a1[38] = "__umoddi3";
  a1[39] = "__umodti3";
  a1[50] = "__negsi2";
  a1[51] = "__negdi2";
  a1[52] = "__clzsi2";
  a1[53] = "__clzdi2";
  a1[54] = "__clzti2";
  a1[55] = "__addsf3";
  a1[56] = "__adddf3";
  a1[57] = "__addxf3";
  a1[58] = "__addtf3";
  a1[59] = "__gcc_qadd";
  a1[60] = "__subsf3";
  a1[61] = "__subdf3";
  a1[62] = "__subxf3";
  a1[63] = "__subtf3";
  a1[64] = "__gcc_qsub";
  a1[65] = "__mulsf3";
  a1[66] = "__muldf3";
  a1[67] = "__mulxf3";
  a1[68] = "__multf3";
  a1[69] = "__gcc_qmul";
  a1[70] = "__divsf3";
  a1[71] = "__divdf3";
  a1[72] = "__divxf3";
  a1[73] = "__divtf3";
  a1[74] = "__gcc_qdiv";
  a1[75] = "fmodf";
  a1[76] = "fmod";
  a1[80] = "fmaf";
  a1[77] = "fmodl";
  a1[78] = "fmodl";
  a1[79] = "fmodl";
  a1[81] = "fma";
  a1[85] = "__powisf2";
  a1[82] = "fmal";
  a1[83] = "fmal";
  a1[84] = "fmal";
  a1[86] = "__powidf2";
  a1[87] = "__powixf2";
  a1[88] = "__powitf2";
  a1[89] = "__powitf2";
  a1[90] = "sqrtf";
  a1[91] = "sqrt";
  a1[92] = "sqrtl";
  a1[93] = "sqrtl";
  a1[94] = "sqrtl";
  a1[95] = "cbrtf";
  a1[96] = "cbrt";
  a1[97] = "cbrtl";
  a1[98] = "cbrtl";
  a1[99] = "cbrtl";
  a1[100] = "logf";
  a1[101] = "log";
  a1[102] = "logl";
  a1[103] = "logl";
  a1[104] = "logl";
  a1[105] = "__logf_finite";
  a1[106] = "__log_finite";
  a1[107] = "__logl_finite";
  a1[108] = "__logl_finite";
  a1[109] = "__logl_finite";
  a1[110] = "log2f";
  a1[111] = "log2";
  a1[112] = "log2l";
  a1[113] = "log2l";
  a1[114] = "log2l";
  a1[115] = "__log2f_finite";
  a1[116] = "__log2_finite";
  a1[117] = "__log2l_finite";
  a1[118] = "__log2l_finite";
  a1[119] = "__log2l_finite";
  a1[120] = "log10f";
  a1[121] = "log10";
  a1[122] = "log10l";
  a1[123] = "log10l";
  a1[124] = "log10l";
  a1[125] = "__log10f_finite";
  a1[126] = "__log10_finite";
  a1[127] = "__log10l_finite";
  a1[128] = "__log10l_finite";
  a1[129] = "__log10l_finite";
  a1[130] = "expf";
  a1[131] = "exp";
  a1[132] = "expl";
  a1[133] = "expl";
  a1[134] = "expl";
  a1[135] = "__expf_finite";
  a1[136] = "__exp_finite";
  a1[137] = "__expl_finite";
  a1[138] = "__expl_finite";
  a1[139] = "__expl_finite";
  a1[140] = "exp2f";
  a1[141] = "exp2";
  a1[142] = "exp2l";
  a1[143] = "exp2l";
  a1[144] = "exp2l";
  a1[145] = "__exp2f_finite";
  a1[146] = "__exp2_finite";
  a1[147] = "__exp2l_finite";
  a1[148] = "__exp2l_finite";
  a1[149] = "__exp2l_finite";
  a1[150] = "exp10f";
  a1[151] = "exp10";
  a1[152] = "exp10l";
  a1[153] = "exp10l";
  a1[154] = "exp10l";
  a1[155] = "sinf";
  a1[156] = "sin";
  a1[157] = "sinl";
  a1[158] = "sinl";
  a1[159] = "sinl";
  a1[160] = "cosf";
  a1[161] = "cos";
  a1[162] = "cosl";
  a1[163] = "cosl";
  a1[164] = "cosl";
  a1[165] = "tanf";
  a1[166] = "tan";
  a1[167] = "tanl";
  a1[168] = "tanl";
  a1[169] = "tanl";
  a1[170] = "sinhf";
  a1[171] = "sinh";
  a1[172] = "sinhl";
  a1[173] = "sinhl";
  a1[174] = "sinhl";
  a1[175] = "coshf";
  a1[176] = "cosh";
  a1[177] = "coshl";
  a1[178] = "coshl";
  a1[179] = "coshl";
  a1[180] = "tanhf";
  a1[181] = "tanh";
  a1[182] = "tanhl";
  a1[183] = "tanhl";
  a1[184] = "tanhl";
  a1[185] = "asinf";
  a1[186] = "asin";
  a1[187] = "asinl";
  a1[188] = "asinl";
  a1[189] = "asinl";
  a1[190] = "acosf";
  a1[191] = "acos";
  a1[192] = "acosl";
  a1[193] = "acosl";
  a1[194] = "acosl";
  a1[195] = "atanf";
  a1[196] = "atan";
  a1[197] = "atanl";
  a1[198] = "atanl";
  a1[199] = "atanl";
  a1[200] = "atan2f";
  a1[201] = "atan2";
  a1[202] = "atan2l";
  a1[203] = "atan2l";
  a1[204] = "atan2l";
  a1[212] = "powf";
  a1[213] = "pow";
  a1[214] = "powl";
  a1[215] = "powl";
  a1[216] = "powl";
  a1[217] = "__powf_finite";
  a1[218] = "__pow_finite";
  a1[219] = "__powl_finite";
  a1[220] = "__powl_finite";
  a1[221] = "__powl_finite";
  a1[222] = "ceilf";
  a1[223] = "ceil";
  a1[224] = "ceill";
  a1[225] = "ceill";
  a1[226] = "ceill";
  a1[227] = "truncf";
  a1[228] = "trunc";
  a1[229] = "truncl";
  a1[230] = "truncl";
  a1[231] = "truncl";
  a1[232] = "rintf";
  a1[233] = "rint";
  a1[234] = "rintl";
  a1[235] = "rintl";
  a1[236] = "rintl";
  a1[237] = "nearbyintf";
  a1[238] = "nearbyint";
  a1[239] = "nearbyintl";
  a1[240] = "nearbyintl";
  a1[241] = "nearbyintl";
  a1[242] = "roundf";
  a1[243] = "round";
  a1[244] = "roundl";
  a1[245] = "roundl";
  a1[246] = "roundl";
  a1[247] = "roundevenf";
  a1[248] = "roundeven";
  a1[249] = "roundevenl";
  a1[250] = "roundevenl";
  a1[251] = "roundevenl";
  a1[252] = "floorf";
  a1[253] = "floor";
  a1[254] = "floorl";
  a1[255] = "floorl";
  a1[256] = "floorl";
  a1[257] = "copysignf";
  a1[258] = "copysign";
  a1[259] = "copysignl";
  a1[260] = "copysignl";
  a1[261] = "copysignl";
  a1[262] = "fminf";
  a1[263] = "fmin";
  a1[264] = "fminl";
  a1[265] = "fminl";
  a1[266] = "fminl";
  a1[267] = "fmaxf";
  a1[268] = "fmax";
  a1[269] = "fmaxl";
  a1[270] = "fmaxl";
  a1[271] = "fmaxl";
  a1[272] = "fminimumf";
  a1[273] = "fminimum";
  a1[278] = "fmaximum";
  a1[277] = "fmaximumf";
  a1[283] = "fminimum_num";
  a1[274] = "fminimuml";
  a1[275] = "fminimuml";
  a1[276] = "fminimuml";
  a1[282] = "fminimum_numf";
  a1[284] = "fminimum_numl";
  a1[285] = "fminimum_numl";
  a1[286] = "fminimum_numl";
  a1[279] = "fmaximuml";
  a1[280] = "fmaximuml";
  a1[287] = "fmaximum_numf";
  a1[288] = "fmaximum_num";
  a1[281] = "fmaximum_numl";
  a1[289] = "fmaximum_numl";
  a1[290] = "fmaximum_numl";
  a1[291] = "fmaximum_numl";
  a1[292] = "lroundf";
  a1[293] = "lround";
  a1[294] = "lroundl";
  a1[295] = "lroundl";
  a1[296] = "lroundl";
  a1[297] = "llroundf";
  a1[298] = "llround";
  a1[299] = "llroundl";
  a1[300] = "llroundl";
  a1[301] = "llroundl";
  a1[302] = "lrintf";
  a1[303] = "lrint";
  a1[304] = "lrintl";
  a1[305] = "lrintl";
  a1[306] = "lrintl";
  a1[307] = "llrintf";
  a1[308] = "llrint";
  a1[312] = "ldexpf";
  a1[313] = "ldexp";
  a1[317] = "frexpf";
  a1[318] = "frexp";
  a1[322] = "sincospif";
  a1[323] = "sincospi";
  a1[309] = "llrintl";
  a1[310] = "llrintl";
  a1[311] = "llrintl";
  a1[327] = "modff";
  a1[328] = "modf";
  a1[314] = "ldexpl";
  a1[315] = "ldexpl";
  a1[316] = "ldexpl";
  a1[332] = "fegetenv";
  a1[333] = "fesetenv";
  a1[319] = "frexpl";
  a1[320] = "frexpl";
  a1[321] = "frexpl";
  a1[334] = "fegetmode";
  a1[335] = "fesetmode";
  a1[324] = "sincospil";
  a1[325] = "sincospil";
  a1[326] = "sincospil";
  a1[336] = "__extendbfsf2";
  a1[337] = "__gcc_stoq";
  a1[329] = "modfl";
  a1[330] = "modfl";
  a1[331] = "modfl";
  a1[338] = "__gcc_dtoq";
  a1[339] = "__extendxftf2";
  a1[340] = "__extenddftf2";
  a1[341] = "__extendsftf2";
  a1[342] = "__extendhftf2";
  a1[343] = "__extendhfxf2";
  a1[344] = "__extendsfdf2";
  a1[345] = "__extendhfdf2";
  a1[346] = "__extendhfsf2";
  a1[347] = "__truncsfhf2";
  a1[348] = "__truncdfhf2";
  a1[349] = "__truncxfhf2";
  a1[352] = "__truncsfbf2";
  a1[353] = "__truncdfbf2";
  a1[354] = "__truncxfbf2";
  a1[355] = "__trunctfbf2";
  a1[356] = "__truncdfsf2";
  a1[357] = "__truncxfsf2";
  a1[358] = "__trunctfsf2";
  a1[359] = "__gcc_qtos";
  a1[360] = "__truncxfdf2";
  a1[361] = "__trunctfdf2";
  a1[362] = "__gcc_qtod";
  a1[363] = "__trunctfxf2";
  a1[364] = "__fixhfsi";
  a1[365] = "__fixhfdi";
  a1[366] = "__fixhfti";
  a1[367] = "__fixsfsi";
  a1[368] = "__fixsfdi";
  a1[369] = "__fixsfti";
  a1[370] = "__fixdfsi";
  a1[371] = "__fixdfdi";
  a1[350] = "__trunctfhf2";
  a1[351] = "__trunctfhf2";
  a1[372] = "__fixdfti";
  a1[373] = "__fixxfsi";
  a1[374] = "__fixxfdi";
  a1[375] = "__fixxfti";
  a1[376] = "__fixtfsi";
  a1[377] = "__fixtfdi";
  a1[380] = "__fixtfdi";
  a1[379] = "__gcc_qtou";
  a1[382] = "__fixunshfsi";
  a1[383] = "__fixunshfdi";
  a1[384] = "__fixunshfti";
  a1[385] = "__fixunssfsi";
  a1[386] = "__fixunssfdi";
  a1[378] = "__fixtfti";
  a1[381] = "__fixtfti";
  a1[387] = "__fixunssfti";
  a1[388] = "__fixunsdfsi";
  a1[389] = "__fixunsdfdi";
  a1[390] = "__fixunsdfti";
  a1[391] = "__fixunsxfsi";
  a1[392] = "__fixunsxfdi";
  a1[393] = "__fixunsxfti";
  a1[396] = "__fixunstfti";
  a1[399] = "__fixunstfti";
  a1[394] = "__fixunstfsi";
  a1[395] = "__fixunstfdi";
  a1[397] = "__fixunstfsi";
  a1[398] = "__fixunstfdi";
  a1[400] = "__floatsihf";
  a1[401] = "__floatsisf";
  a1[402] = "__floatsidf";
  a1[403] = "__floatsixf";
  a1[404] = "__floatsitf";
  a1[405] = "__gcc_itoq";
  a1[406] = "__floatdibf";
  a1[409] = "__floatdidf";
  a1[411] = "__floatditf";
  a1[412] = "__floatditf";
  a1[414] = "__floattisf";
  a1[408] = "__floatdisf";
  a1[417] = "__floattitf";
  a1[418] = "__floattitf";
  a1[407] = "__floatdihf";
  a1[413] = "__floattihf";
  a1[419] = "__floatunsihf";
  a1[410] = "__floatdixf";
  a1[416] = "__floattixf";
  a1[422] = "__floatunsixf";
  a1[415] = "__floattidf";
  a1[421] = "__floatunsidf";
  a1[425] = "__floatundibf";
  a1[420] = "__floatunsisf";
  a1[424] = "__gcc_utoq";
  a1[428] = "__floatundidf";
  a1[423] = "__floatunsitf";
  a1[427] = "__floatundisf";
  a1[430] = "__floatunditf";
  a1[431] = "__floatunditf";
  a1[426] = "__floatundihf";
  a1[432] = "__floatuntihf";
  a1[433] = "__floatuntisf";
  a1[429] = "__floatundixf";
  a1[435] = "__floatuntixf";
  a1[436] = "__floatuntitf";
  a1[437] = "__floatuntitf";
  a1[434] = "__floatuntidf";
  a1[438] = "__extendkftf2";
  a1[440] = "__eqsf2";
  a1[439] = "__trunctfkf2";
  a1[441] = "__eqdf2";
  a1[443] = "__gcc_qeq";
  a1[442] = "__eqtf2";
  a1[444] = "__nesf2";
  a1[446] = "__netf2";
  a1[445] = "__nedf2";
  a1[447] = "__gcc_qne";
  a1[449] = "__gedf2";
  a1[448] = "__gesf2";
  a1[450] = "__getf2";
  a1[452] = "__ltsf2";
  a1[451] = "__gcc_qge";
  a1[453] = "__ltdf2";
  a1[455] = "__gcc_qlt";
  a1[454] = "__lttf2";
  a1[456] = "__lesf2";
  a1[458] = "__letf2";
  a1[457] = "__ledf2";
  a1[459] = "__gcc_qle";
  a1[461] = "__gtdf2";
  a1[460] = "__gtsf2";
  a1[462] = "__gttf2";
  a1[464] = "__unordsf2";
  a1[463] = "__gcc_qgt";
  a1[465] = "__unorddf2";
  a1[467] = "__gcc_qunord";
  a1[466] = "__unordtf2";
  a1[468] = "memcpy";
  a1[470] = "memset";
  a1[469] = "memmove";
  a1[471] = "calloc";
  a1[474] = "__llvm_memcpy_element_unordered_atomic_2";
  a1[473] = "__llvm_memcpy_element_unordered_atomic_1";
  a1[475] = "__llvm_memcpy_element_unordered_atomic_4";
  a1[477] = "__llvm_memcpy_element_unordered_atomic_16";
  a1[476] = "__llvm_memcpy_element_unordered_atomic_8";
  a1[478] = "__llvm_memmove_element_unordered_atomic_1";
  a1[480] = "__llvm_memmove_element_unordered_atomic_4";
  a1[479] = "__llvm_memmove_element_unordered_atomic_2";
  a1[481] = "__llvm_memmove_element_unordered_atomic_8";
  a1[483] = "__llvm_memset_element_unordered_atomic_1";
  a1[482] = "__llvm_memmove_element_unordered_atomic_16";
  a1[484] = "__llvm_memset_element_unordered_atomic_2";
  a1[486] = "__llvm_memset_element_unordered_atomic_8";
  a1[485] = "__llvm_memset_element_unordered_atomic_4";
  a1[487] = "__llvm_memset_element_unordered_atomic_16";
  a1[489] = "__cxa_end_cleanup";
  a1[488] = "_Unwind_Resume";
  a1[490] = "__sync_val_compare_and_swap_1";
  a1[492] = "__sync_val_compare_and_swap_4";
  a1[491] = "__sync_val_compare_and_swap_2";
  a1[493] = "__sync_val_compare_and_swap_8";
  a1[495] = "__sync_lock_test_and_set_1";
  a1[494] = "__sync_val_compare_and_swap_16";
  a1[496] = "__sync_lock_test_and_set_2";
  a1[498] = "__sync_lock_test_and_set_8";
  a1[497] = "__sync_lock_test_and_set_4";
  a1[499] = "__sync_lock_test_and_set_16";
  a1[501] = "__sync_fetch_and_add_2";
  a1[500] = "__sync_fetch_and_add_1";
  a1[502] = "__sync_fetch_and_add_4";
  a1[504] = "__sync_fetch_and_add_16";
  a1[503] = "__sync_fetch_and_add_8";
  a1[505] = "__sync_fetch_and_sub_1";
  a1[507] = "__sync_fetch_and_sub_4";
  a1[506] = "__sync_fetch_and_sub_2";
  a1[508] = "__sync_fetch_and_sub_8";
  a1[510] = "__sync_fetch_and_and_1";
  a1[509] = "__sync_fetch_and_sub_16";
  a1[511] = "__sync_fetch_and_and_2";
  a1[513] = "__sync_fetch_and_and_8";
  a1[512] = "__sync_fetch_and_and_4";
  a1[514] = "__sync_fetch_and_and_16";
  a1[516] = "__sync_fetch_and_or_2";
  a1[515] = "__sync_fetch_and_or_1";
  a1[517] = "__sync_fetch_and_or_4";
  a1[519] = "__sync_fetch_and_or_16";
  a1[518] = "__sync_fetch_and_or_8";
  a1[520] = "__sync_fetch_and_xor_1";
  a1[522] = "__sync_fetch_and_xor_4";
  a1[521] = "__sync_fetch_and_xor_2";
  a1[523] = "__sync_fetch_and_xor_8";
  a1[525] = "__sync_fetch_and_nand_1";
  a1[524] = "__sync_fetch_and_xor_16";
  a1[526] = "__sync_fetch_and_nand_2";
  a1[528] = "__sync_fetch_and_nand_8";
  a1[527] = "__sync_fetch_and_nand_4";
  a1[529] = "__sync_fetch_and_nand_16";
  a1[531] = "__sync_fetch_and_max_2";
  a1[530] = "__sync_fetch_and_max_1";
  a1[532] = "__sync_fetch_and_max_4";
  a1[534] = "__sync_fetch_and_max_16";
  a1[533] = "__sync_fetch_and_max_8";
  a1[535] = "__sync_fetch_and_umax_1";
  a1[537] = "__sync_fetch_and_umax_4";
  a1[536] = "__sync_fetch_and_umax_2";
  a1[538] = "__sync_fetch_and_umax_8";
  a1[540] = "__sync_fetch_and_min_1";
  a1[539] = "__sync_fetch_and_umax_16";
  a1[541] = "__sync_fetch_and_min_2";
  a1[543] = "__sync_fetch_and_min_8";
  a1[542] = "__sync_fetch_and_min_4";
  a1[544] = "__sync_fetch_and_min_16";
  a1[546] = "__sync_fetch_and_umin_2";
  a1[545] = "__sync_fetch_and_umin_1";
  a1[547] = "__sync_fetch_and_umin_4";
  a1[549] = "__sync_fetch_and_umin_16";
  a1[548] = "__sync_fetch_and_umin_8";
  a1[550] = "__atomic_load";
  a1[552] = "__atomic_load_2";
  a1[551] = "__atomic_load_1";
  a1[553] = "__atomic_load_4";
  a1[555] = "__atomic_load_16";
  a1[554] = "__atomic_load_8";
  a1[556] = "__atomic_store";
  a1[558] = "__atomic_store_2";
  a1[557] = "__atomic_store_1";
  a1[559] = "__atomic_store_4";
  a1[561] = "__atomic_store_16";
  a1[560] = "__atomic_store_8";
  a1[562] = "__atomic_exchange";
  a1[564] = "__atomic_exchange_2";
  a1[563] = "__atomic_exchange_1";
  a1[565] = "__atomic_exchange_4";
  a1[567] = "__atomic_exchange_16";
  a1[566] = "__atomic_exchange_8";
  a1[568] = "__atomic_compare_exchange";
  a1[570] = "__atomic_compare_exchange_2";
  a1[569] = "__atomic_compare_exchange_1";
  a1[571] = "__atomic_compare_exchange_4";
  a1[573] = "__atomic_compare_exchange_16";
  a1[572] = "__atomic_compare_exchange_8";
  a1[574] = "__atomic_fetch_add_1";
  a1[576] = "__atomic_fetch_add_4";
  a1[575] = "__atomic_fetch_add_2";
  a1[577] = "__atomic_fetch_add_8";
  a1[579] = "__atomic_fetch_sub_1";
  a1[578] = "__atomic_fetch_add_16";
  a1[580] = "__atomic_fetch_sub_2";
  a1[582] = "__atomic_fetch_sub_8";
  a1[581] = "__atomic_fetch_sub_4";
  a1[583] = "__atomic_fetch_sub_16";
  a1[585] = "__atomic_fetch_and_2";
  a1[584] = "__atomic_fetch_and_1";
  a1[586] = "__atomic_fetch_and_4";
  a1[588] = "__atomic_fetch_and_16";
  a1[587] = "__atomic_fetch_and_8";
  a1[589] = "__atomic_fetch_or_1";
  a1[591] = "__atomic_fetch_or_4";
  a1[590] = "__atomic_fetch_or_2";
  a1[592] = "__atomic_fetch_or_8";
  a1[594] = "__atomic_fetch_xor_1";
  a1[593] = "__atomic_fetch_or_16";
  a1[595] = "__atomic_fetch_xor_2";
  a1[597] = "__atomic_fetch_xor_8";
  a1[596] = "__atomic_fetch_xor_4";
  a1[598] = "__atomic_fetch_xor_16";
  a1[600] = "__atomic_fetch_nand_2";
  a1[599] = "__atomic_fetch_nand_1";
  a1[601] = "__atomic_fetch_nand_4";
  a1[603] = "__atomic_fetch_nand_16";
  a1[602] = "__atomic_fetch_nand_8";
  a1[724] = "__stack_chk_fail";
  a1[727] = "__clear_cache";
  a1[728] = "__riscv_flush_icache";
  a1[725] = "__llvm_deoptimize";
  a1[730] = 0;
  *(_QWORD *)((char *)a1 + 8748) = 0;
  memset((void *)(v2 & 0xFFFFFFFFFFFFFFF8LL), 0, 8 * (((unsigned int)a1 - (v2 & 0xFFFFFFF8) + 8756) >> 3));
  v5 = a2[8];
  if ( v5 == 39 )
  {
    if ( (unsigned int)(a2[12] - 1) > 0xB )
      goto LABEL_4;
    a1[78] = "fmodf128";
    a1[83] = "fmaf128";
    a1[93] = "sqrtf128";
    a1[98] = "cbrtf128";
    a1[103] = "logf128";
    a1[108] = "__logf128_finite";
    a1[113] = "log2f128";
    a1[118] = "__log2f128_finite";
    a1[123] = "log10f128";
    a1[128] = "__log10f128_finite";
    a1[133] = "expf128";
    a1[138] = "__expf128_finite";
    a1[143] = "exp2f128";
    a1[148] = "__exp2f128_finite";
    a1[153] = "exp10f128";
    a1[158] = "sinf128";
    a1[163] = "cosf128";
    a1[168] = "tanf128";
    a1[208] = "sincosf128";
    a1[188] = "asinf128";
    a1[193] = "acosf128";
    a1[198] = "atanf128";
    a1[203] = "atan2f128";
    a1[173] = "sinhf128";
    a1[178] = "coshf128";
    a1[183] = "tanhf128";
    a1[215] = "powf128";
    a1[220] = "__powf128_finite";
    a1[225] = "ceilf128";
    a1[230] = "truncf128";
    a1[235] = "rintf128";
    a1[240] = "nearbyintf128";
    a1[245] = "roundf128";
    a1[250] = "roundevenf128";
    a1[255] = "floorf128";
    a1[260] = "copysignf128";
    a1[265] = "fminf128";
    a1[270] = "fmaxf128";
    a1[295] = "lroundf128";
    a1[300] = "llroundf128";
    a1[305] = "lrintf128";
    a1[310] = "llrintf128";
    a1[315] = "ldexpf128";
    a1[320] = "frexpf128";
    v5 = a2[8];
  }
  if ( (unsigned int)(v5 - 22) <= 3 )
  {
    a1[58] = "__addkf3";
    a1[63] = "__subkf3";
    a1[68] = "__mulkf3";
    a1[73] = "__divkf3";
    a1[88] = "__powikf2";
    a1[341] = "__extendsfkf2";
    a1[340] = "__extenddfkf2";
    a1[358] = "__trunckfsf2";
    a1[361] = "__trunckfdf2";
    a1[376] = "__fixkfsi";
    a1[377] = "__fixkfdi";
    a1[378] = "__fixkfti";
    a1[394] = "__fixunskfsi";
    a1[395] = "__fixunskfdi";
    a1[396] = "__fixunskfti";
    a1[404] = "__floatsikf";
    a1[411] = "__floatdikf";
    a1[417] = "__floattikf";
    a1[423] = "__floatunsikf";
    a1[430] = "__floatundikf";
    a1[436] = "__floatuntikf";
    a1[442] = "__eqkf2";
    a1[446] = "__nekf2";
    a1[450] = "__gekf2";
    a1[454] = "__ltkf2";
    a1[458] = "__lekf2";
    a1[462] = "__gtkf2";
    a1[466] = "__unordkf2";
  }
LABEL_4:
  v6 = (unsigned int)a2[11];
  if ( (_DWORD)v6 == 1 )
  {
    v19 = a2[8];
    if ( v19 != 5 )
    {
      if ( v19 > 5 )
      {
        if ( v19 - 38 > 1 )
        {
          if ( v19 == 38 )
            goto LABEL_22;
          goto LABEL_61;
        }
        goto LABEL_74;
      }
      if ( v19 != 3 )
        goto LABEL_61;
    }
LABEL_55:
    a1[472] = "bzero";
LABEL_56:
    v20 = a2[8];
    LODWORD(v6) = a2[11];
    goto LABEL_57;
  }
  if ( (unsigned int)v6 > 0x1F )
    goto LABEL_22;
  v7 = 3623879200LL;
  if ( !_bittest64(&v7, v6) )
  {
    if ( (_DWORD)v6 != 29 )
      goto LABEL_22;
    goto LABEL_21;
  }
  v8 = a2[8];
  if ( v8 > 0x27 )
    goto LABEL_11;
  if ( v8 <= 0x25 )
  {
    if ( ((v8 - 3) & 0xFFFFFFFD) != 0 )
      goto LABEL_11;
    goto LABEL_55;
  }
  if ( (_DWORD)v6 != 9 )
  {
    if ( v8 == 38 )
      goto LABEL_18;
LABEL_11:
    if ( (v6 & 0xFFFFFFF7) != 1 )
    {
      if ( (_DWORD)v6 != 27 && (_DWORD)v6 != 5 || (unsigned int)sub_CC78E0((__int64)a2) > 6 )
      {
LABEL_14:
        a1[210] = "__sincosf_stret";
        a1[211] = "__sincos_stret";
        if ( a2[9] == 26 )
          a1[835] = 0x4400000044LL;
      }
LABEL_16:
      LODWORD(v6) = a2[11];
      goto LABEL_17;
    }
LABEL_61:
    if ( !sub_CC8200((__int64)a2, 0xAu, 9, 0) && sub_CC7F40((__int64)a2) )
      goto LABEL_14;
    goto LABEL_16;
  }
LABEL_74:
  if ( sub_CC8200((__int64)a2, 0xAu, 6, 0) )
    goto LABEL_56;
  a1[472] = "__bzero";
  v20 = a2[8];
  LODWORD(v6) = a2[11];
LABEL_57:
  if ( v20 != 38 )
    goto LABEL_11;
LABEL_17:
  if ( (_DWORD)v6 == 9 )
  {
    if ( !sub_CC8200((__int64)a2, 0xAu, 9, 0) )
      goto LABEL_21;
LABEL_68:
    a1[150] = 0;
    a1[151] = 0;
    goto LABEL_22;
  }
LABEL_18:
  if ( (unsigned int)v6 <= 9 )
  {
    if ( (_DWORD)v6 != 5 )
      goto LABEL_22;
    if ( (unsigned int)sub_CC78E0((__int64)a2) > 6 )
      goto LABEL_21;
    goto LABEL_68;
  }
  if ( (unsigned int)v6 > 0x1C )
  {
    if ( (unsigned int)(v6 - 30) > 1 )
      goto LABEL_22;
    goto LABEL_21;
  }
  if ( (unsigned int)v6 > 0x1A )
  {
LABEL_21:
    a1[150] = "__exp10f";
    a1[151] = "__exp10";
  }
LABEL_22:
  v9 = a2[12];
  if ( (unsigned int)(v9 - 1) <= 0xB || (v10 = a2[11], v10 == 4) )
  {
LABEL_26:
    a1[205] = "sincosf";
    a1[206] = "sincos";
    a1[207] = "sincosl";
    a1[208] = "sincosl";
    a1[209] = "sincosl";
LABEL_27:
    v10 = a2[11];
    goto LABEL_28;
  }
  if ( v9 == 17 )
  {
    v11 = sub_CC7810((__int64)a2);
    if ( !sub_CC7F40((__int64)a2) && v11 <= 8 )
      goto LABEL_27;
    goto LABEL_26;
  }
LABEL_28:
  if ( a2[8] == 39 && a2[10] == 3 && (unsigned int)(v10 - 24) <= 1 )
  {
    a1[205] = "sincosf";
    a1[206] = "sincos";
    v10 = a2[11];
  }
  if ( v10 == 11 )
  {
    a1[724] = 0;
    v10 = a2[11];
  }
  if ( v10 == 14 )
  {
    v21 = a2[12];
    if ( v21 != 29 && v21 != 1 )
    {
      a1[312] = 0;
      a1[314] = 0;
      a1[315] = 0;
      a1[316] = 0;
      a1[317] = 0;
      a1[319] = 0;
      a1[320] = 0;
      a1[321] = 0;
    }
  }
  v12 = a2[8];
  if ( (unsigned int)(v12 - 26) <= 1 )
  {
    for ( i = 0; i != 729; ++i )
    {
      if ( (unsigned int)(i - 550) > 0x35 )
        a1[i] = 0;
    }
    v12 = a2[8];
  }
  v14 = v12 - 42;
  v15 = 0;
  if ( v14 <= 1 )
  {
    do
    {
      if ( (unsigned int)(v15 - 550) > 0x35 )
        a1[v15] = 0;
      ++v15;
    }
    while ( v15 != 729 );
  }
  if ( a2[11] == 14 )
  {
    v18 = a2[12];
    if ( v18 - 27 <= 1 || v18 <= 1 )
    {
      a1[85] = 0;
      a1[86] = 0;
    }
  }
  v16 = a2[8];
  if ( v16 == 7 )
  {
    a1[20] = 0;
    a1[21] = 0;
    a1[22] = 0;
    a1[25] = 0;
    a1[26] = 0;
    a1[27] = 0;
    a1[30] = 0;
    a1[31] = 0;
    a1[32] = 0;
    a1[35] = 0;
    a1[36] = 0;
    a1[37] = 0;
    v16 = a2[8];
  }
  v17 = v16 - 56;
  if ( v17 > 1 )
  {
    LOBYTE(v17) = sub_CC7F60((__int64)a2);
    if ( (_BYTE)v17 )
    {
      a1[3] = 0;
      a1[7] = 0;
      a1[11] = 0;
      a1[16] = 0;
      a1[18] = 0;
    }
    a1[19] = 0;
  }
  return v17;
}
