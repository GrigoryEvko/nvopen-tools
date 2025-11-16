// Function: sub_C09360
// Address: 0xc09360
//
__int64 __fastcall sub_C09360(__int64 *a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  bool v6; // zf
  __int64 *v7; // rbx
  int v8; // r13d
  int v9; // eax
  unsigned int v10; // r13d
  _QWORD v13[2]; // [rsp+10h] [rbp-940h] BYREF
  _BYTE v14[112]; // [rsp+20h] [rbp-930h] BYREF
  _QWORD *v15; // [rsp+90h] [rbp-8C0h]
  _QWORD *v16; // [rsp+98h] [rbp-8B8h]
  __int64 v17; // [rsp+A0h] [rbp-8B0h]
  __int16 v18; // [rsp+A8h] [rbp-8A8h]
  bool v19; // [rsp+AAh] [rbp-8A6h]
  char *v20; // [rsp+B0h] [rbp-8A0h]
  __int64 v21; // [rsp+B8h] [rbp-898h]
  char v22; // [rsp+C0h] [rbp-890h] BYREF
  char *v23; // [rsp+C8h] [rbp-888h]
  __int64 v24; // [rsp+D0h] [rbp-880h]
  char v25; // [rsp+D8h] [rbp-878h] BYREF
  __int64 v26; // [rsp+110h] [rbp-840h]
  __int64 v27; // [rsp+118h] [rbp-838h]
  char v28; // [rsp+120h] [rbp-830h]
  __int64 v29; // [rsp+124h] [rbp-82Ch]
  __int64 v30; // [rsp+130h] [rbp-820h]
  char *v31; // [rsp+138h] [rbp-818h]
  __int64 v32; // [rsp+140h] [rbp-810h]
  int v33; // [rsp+148h] [rbp-808h]
  char v34; // [rsp+14Ch] [rbp-804h]
  char v35; // [rsp+150h] [rbp-800h] BYREF
  __int64 v36; // [rsp+1D0h] [rbp-780h]
  char *v37; // [rsp+1D8h] [rbp-778h]
  __int64 v38; // [rsp+1E0h] [rbp-770h]
  int v39; // [rsp+1E8h] [rbp-768h]
  char v40; // [rsp+1ECh] [rbp-764h]
  char v41; // [rsp+1F0h] [rbp-760h] BYREF
  __int64 v42; // [rsp+2F0h] [rbp-660h]
  __int64 v43; // [rsp+2F8h] [rbp-658h]
  __int64 v44; // [rsp+300h] [rbp-650h]
  int v45; // [rsp+308h] [rbp-648h]
  __int64 v46; // [rsp+310h] [rbp-640h]
  char *v47; // [rsp+318h] [rbp-638h]
  __int64 v48; // [rsp+320h] [rbp-630h]
  int v49; // [rsp+328h] [rbp-628h]
  char v50; // [rsp+32Ch] [rbp-624h]
  char v51; // [rsp+330h] [rbp-620h] BYREF
  __int64 v52; // [rsp+340h] [rbp-610h]
  __int16 v53; // [rsp+348h] [rbp-608h]
  __int64 v54; // [rsp+350h] [rbp-600h]
  __int64 v55; // [rsp+358h] [rbp-5F8h]
  __int64 v56; // [rsp+360h] [rbp-5F0h]
  int v57; // [rsp+368h] [rbp-5E8h]
  __int64 v58; // [rsp+370h] [rbp-5E0h]
  __int64 v59; // [rsp+378h] [rbp-5D8h]
  __int64 v60; // [rsp+380h] [rbp-5D0h]
  int v61; // [rsp+388h] [rbp-5C8h]
  _QWORD *v62; // [rsp+390h] [rbp-5C0h]
  __int64 v63; // [rsp+398h] [rbp-5B8h]
  _QWORD v64[3]; // [rsp+3A0h] [rbp-5B0h] BYREF
  int v65; // [rsp+3B8h] [rbp-598h]
  __int64 v66; // [rsp+3C0h] [rbp-590h]
  char *v67; // [rsp+3C8h] [rbp-588h]
  __int64 v68; // [rsp+3D0h] [rbp-580h]
  int v69; // [rsp+3D8h] [rbp-578h]
  char v70; // [rsp+3DCh] [rbp-574h]
  char v71; // [rsp+3E0h] [rbp-570h] BYREF
  char *v72; // [rsp+4E0h] [rbp-470h]
  __int64 v73; // [rsp+4E8h] [rbp-468h]
  char v74; // [rsp+4F0h] [rbp-460h] BYREF
  __int64 v75; // [rsp+510h] [rbp-440h]
  char *v76; // [rsp+518h] [rbp-438h]
  __int64 v77; // [rsp+520h] [rbp-430h]
  int v78; // [rsp+528h] [rbp-428h]
  char v79; // [rsp+52Ch] [rbp-424h]
  char v80; // [rsp+530h] [rbp-420h] BYREF
  __int64 v81; // [rsp+630h] [rbp-320h]
  char *v82; // [rsp+638h] [rbp-318h]
  __int64 v83; // [rsp+640h] [rbp-310h]
  int v84; // [rsp+648h] [rbp-308h]
  char v85; // [rsp+64Ch] [rbp-304h]
  char v86; // [rsp+650h] [rbp-300h] BYREF
  char *v87; // [rsp+750h] [rbp-200h]
  __int64 v88; // [rsp+758h] [rbp-1F8h]
  char v89; // [rsp+760h] [rbp-1F0h] BYREF
  _QWORD *v90; // [rsp+7E0h] [rbp-170h]
  __int64 v91; // [rsp+7E8h] [rbp-168h]
  __int64 v92; // [rsp+7F0h] [rbp-160h]
  __int64 v93; // [rsp+7F8h] [rbp-158h]
  int v94; // [rsp+800h] [rbp-150h]
  __int64 v95; // [rsp+808h] [rbp-148h]
  __int64 v96; // [rsp+810h] [rbp-140h]
  __int64 v97; // [rsp+818h] [rbp-138h]
  int v98; // [rsp+820h] [rbp-130h]
  __int64 v99; // [rsp+840h] [rbp-110h]
  __int64 v100; // [rsp+860h] [rbp-F0h]
  __int64 v101; // [rsp+868h] [rbp-E8h]
  __int64 v102; // [rsp+870h] [rbp-E0h]
  int v103; // [rsp+878h] [rbp-D8h]
  __int64 v104; // [rsp+880h] [rbp-D0h]
  __int64 v105; // [rsp+888h] [rbp-C8h]
  __int64 v106; // [rsp+890h] [rbp-C0h]
  int v107; // [rsp+898h] [rbp-B8h]
  __int64 v108; // [rsp+8A0h] [rbp-B0h]
  __int64 v109; // [rsp+8A8h] [rbp-A8h]
  __int64 v110; // [rsp+8B0h] [rbp-A0h]
  int v111; // [rsp+8C0h] [rbp-90h]
  __int64 v112; // [rsp+8C8h] [rbp-88h]
  __int64 v113; // [rsp+8D0h] [rbp-80h]
  __int64 v114; // [rsp+8D8h] [rbp-78h]
  int v115; // [rsp+8E0h] [rbp-70h]
  char v116; // [rsp+8E8h] [rbp-68h]
  char *v117; // [rsp+8F0h] [rbp-60h]
  __int64 i; // [rsp+8F8h] [rbp-58h]
  char v119; // [rsp+900h] [rbp-50h] BYREF

  v13[0] = a2;
  v4 = (__int64)a1;
  v13[1] = a1;
  sub_A558A0((__int64)v14, (__int64)a1, 1);
  v26 = 0;
  v15 = a1 + 29;
  v16 = a1 + 39;
  v5 = *a1;
  v27 = 0;
  v17 = v5;
  v6 = a3 == 0;
  v18 = 0;
  v20 = &v22;
  v21 = 0x100000000LL;
  v23 = &v25;
  v24 = 0x600000000LL;
  v31 = &v35;
  v37 = &v41;
  v47 = &v51;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v32 = 16;
  v33 = 0;
  v34 = 1;
  v36 = 0;
  v38 = 32;
  v39 = 0;
  v40 = 1;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v48 = 2;
  v49 = 0;
  v50 = 1;
  v52 = 0;
  v53 = 0;
  v62 = v64;
  v67 = &v71;
  v76 = &v80;
  v72 = &v74;
  v82 = &v86;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v63 = 0;
  memset(v64, 0, sizeof(v64));
  v65 = 0;
  v66 = 0;
  v68 = 32;
  v69 = 0;
  v70 = 1;
  v73 = 0x400000000LL;
  v75 = 0;
  v77 = 32;
  v78 = 0;
  v79 = 1;
  v81 = 0;
  v83 = 32;
  v84 = 0;
  v85 = 1;
  v87 = &v89;
  v7 = (__int64 *)a1[4];
  v19 = v6;
  v8 = 0;
  v88 = 0x1000000000LL;
  v90 = v13;
  v91 = 0;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v111 = 2;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = &v119;
  for ( i = 0x400000000LL; a1 + 3 != v7; v8 |= v9 ^ 1 )
  {
    v4 = (__int64)(v7 - 7);
    if ( !v7 )
      v4 = 0;
    v9 = sub_C05FA0(v13, v4);
    v7 = (__int64 *)v7[1];
  }
  v10 = sub_BF3D50((__int64)v13, v4) ^ 1 | v8;
  if ( a3 )
    *a3 = HIBYTE(v18);
  sub_C08A70((__int64)v13, v4);
  return v10;
}
