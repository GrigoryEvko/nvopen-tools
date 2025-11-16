// Function: sub_209B000
// Address: 0x209b000
//
__int64 *__fastcall sub_209B000(__int64 a1, __int64 a2, __m128i a3, __m128i a4, __m128i a5)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rdx
  unsigned __int8 v9; // cl
  unsigned __int64 v10; // rsi
  unsigned __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // rax
  __int64 *v14; // r13
  int v15; // edx
  unsigned __int64 v16; // r15
  __int64 v17; // r14
  unsigned int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdi
  __int64 v25; // r8
  __int64 v26; // rcx
  unsigned int v27; // esi
  __int64 **v28; // rax
  __int64 *v29; // r9
  unsigned int v30; // edx
  __int64 v31; // rax
  __int64 *v32; // rdi
  __int64 *result; // rax
  unsigned __int64 v34; // rdx
  _QWORD *v35; // r14
  unsigned int v36; // r13d
  __int64 v37; // rax
  unsigned int v38; // edx
  unsigned __int8 v39; // al
  __int64 v40; // rdx
  __int64 *v41; // rax
  _QWORD *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // rsi
  int v45; // edx
  __int64 v46; // rax
  __int64 v47; // rsi
  __int64 v48; // r15
  __int64 v49; // rdi
  char v50; // al
  unsigned int v51; // edx
  unsigned int v52; // eax
  __int64 v53; // rdx
  unsigned int v54; // ecx
  __int64 v55; // r14
  unsigned int v56; // edx
  unsigned int v57; // r15d
  __int64 v58; // r13
  unsigned int v59; // esi
  __int64 v60; // r8
  unsigned int v61; // edi
  __int64 *v62; // rdx
  __int64 v63; // rcx
  unsigned __int64 *v64; // rdx
  int v65; // eax
  int v66; // ecx
  int v67; // edx
  int v68; // esi
  __int64 v69; // r9
  unsigned int v70; // ecx
  int v71; // edi
  __int64 v72; // r8
  int v73; // r11d
  __int64 *v74; // r10
  __int64 v75; // r9
  unsigned int v76; // edx
  char v77; // al
  unsigned int v78; // eax
  _QWORD *v79; // r15
  __int64 v80; // rdx
  unsigned int v81; // ecx
  __int64 v82; // rax
  __int64 v83; // rax
  int v84; // eax
  int v85; // r9d
  int v86; // eax
  int v87; // r11d
  int v88; // eax
  int v89; // eax
  __int64 v90; // rdi
  unsigned int v91; // r9d
  unsigned __int64 v92; // rsi
  int v93; // r11d
  unsigned __int64 *v94; // r10
  int v95; // eax
  int v96; // eax
  unsigned __int64 *v97; // r9
  __int64 v98; // rdi
  int v99; // r10d
  __int64 v100; // r11
  unsigned __int64 v101; // rsi
  int v102; // r11d
  __int64 *v103; // r10
  int v104; // eax
  int v105; // edx
  int v106; // ecx
  __int64 v107; // r8
  int v108; // r10d
  __int64 v109; // r13
  __int64 *v110; // r9
  __int64 v111; // rsi
  unsigned int v112; // [rsp+8h] [rbp-118h]
  __int64 v113; // [rsp+10h] [rbp-110h]
  unsigned int v114; // [rsp+18h] [rbp-108h]
  int v115; // [rsp+20h] [rbp-100h]
  __int64 v116; // [rsp+20h] [rbp-100h]
  __int64 v117; // [rsp+28h] [rbp-F8h]
  __int64 *v118; // [rsp+30h] [rbp-F0h]
  _QWORD *v119; // [rsp+30h] [rbp-F0h]
  __int64 v120; // [rsp+38h] [rbp-E8h]
  unsigned int v121; // [rsp+40h] [rbp-E0h]
  __int64 v122; // [rsp+40h] [rbp-E0h]
  __int64 v123; // [rsp+48h] [rbp-D8h]
  int v124; // [rsp+48h] [rbp-D8h]
  unsigned int v125; // [rsp+48h] [rbp-D8h]
  __int64 v126; // [rsp+90h] [rbp-90h] BYREF
  int v127; // [rsp+98h] [rbp-88h]
  char v128[8]; // [rsp+A0h] [rbp-80h] BYREF
  __int64 v129; // [rsp+A8h] [rbp-78h]
  _QWORD v130[4]; // [rsp+B0h] [rbp-70h] BYREF
  __int128 v131; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v132; // [rsp+E0h] [rbp-40h]

  v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v8 = *(_QWORD *)(a2 - 24 * v7);
  v9 = *(_BYTE *)(v8 + 16);
  if ( v9 == 88 )
  {
    v83 = sub_157F120(*(_QWORD *)(v8 + 40));
    v8 = sub_157EBA0(v83);
    v9 = *(_BYTE *)(v8 + 16);
    v7 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  }
  if ( v9 <= 0x17u )
  {
    v10 = 0;
    goto LABEL_6;
  }
  if ( v9 == 78 )
  {
    v34 = v8 | 4;
  }
  else
  {
    v10 = 0;
    if ( v9 != 29 )
    {
LABEL_6:
      v11 = v10 - 24LL * (*(_DWORD *)(v10 + 20) & 0xFFFFFFF);
      goto LABEL_7;
    }
    v34 = v8 & 0xFFFFFFFFFFFFFFFBLL;
  }
  v10 = v34 & 0xFFFFFFFFFFFFFFF8LL;
  v11 = (v34 & 0xFFFFFFFFFFFFFFF8LL) - 24LL * (*(_DWORD *)((v34 & 0xFFFFFFFFFFFFFFF8LL) + 20) & 0xFFFFFFF);
  if ( (v34 & 4) == 0 )
    goto LABEL_6;
LABEL_7:
  v12 = *(_QWORD *)(a2 + 24 * (2 - v7));
  v13 = *(_QWORD **)(v12 + 24);
  if ( *(_DWORD *)(v12 + 32) > 0x40u )
    v13 = (_QWORD *)*v13;
  v14 = *(__int64 **)(v11 + 24LL * (unsigned int)v13);
  v118 = sub_20685E0(a1, v14, a3, a4, a5);
  v115 = v15;
  v16 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v16 + 16) == 88 )
  {
    v82 = sub_157F120(*(_QWORD *)(v16 + 40));
    v16 = sub_157EBA0(v82);
  }
  v17 = *(_QWORD *)(a1 + 712);
  v18 = *(_DWORD *)(v17 + 328);
  if ( !v18 )
  {
    ++*(_QWORD *)(v17 + 304);
    goto LABEL_99;
  }
  v19 = *(_QWORD *)(v17 + 312);
  v20 = (v18 - 1) & (((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9));
  v21 = (__int64 *)(v19 + 72LL * v20);
  v22 = *v21;
  if ( v16 != *v21 )
  {
    v124 = 1;
    v64 = 0;
    while ( v22 != -8 )
    {
      if ( v22 == -16 && !v64 )
        v64 = (unsigned __int64 *)v21;
      v20 = (v18 - 1) & (v124 + v20);
      v21 = (__int64 *)(v19 + 72LL * v20);
      v22 = *v21;
      if ( v16 == *v21 )
        goto LABEL_13;
      ++v124;
    }
    if ( !v64 )
      v64 = (unsigned __int64 *)v21;
    v65 = *(_DWORD *)(v17 + 320);
    ++*(_QWORD *)(v17 + 304);
    v66 = v65 + 1;
    if ( 4 * (v65 + 1) < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(v17 + 324) - v66 > v18 >> 3 )
      {
LABEL_54:
        *(_DWORD *)(v17 + 320) = v66;
        if ( *v64 != -8 )
          --*(_DWORD *)(v17 + 324);
        *v64 = v16;
        a3.m128i_i64[0] = 0;
        v26 = 0;
        v25 = 0;
        *(_OWORD *)(v64 + 1) = 0;
        *(_OWORD *)(v64 + 3) = 0;
        *(_OWORD *)(v64 + 5) = 0;
        *(_OWORD *)(v64 + 7) = 0;
        goto LABEL_57;
      }
      v125 = ((unsigned int)v16 >> 4) ^ ((unsigned int)v16 >> 9);
      sub_209A280(v17 + 304, v18);
      v95 = *(_DWORD *)(v17 + 328);
      if ( v95 )
      {
        v96 = v95 - 1;
        v97 = 0;
        v98 = *(_QWORD *)(v17 + 312);
        v99 = 1;
        LODWORD(v100) = v96 & v125;
        v66 = *(_DWORD *)(v17 + 320) + 1;
        v64 = (unsigned __int64 *)(v98 + 72LL * (v96 & v125));
        v101 = *v64;
        if ( v16 != *v64 )
        {
          while ( v101 != -8 )
          {
            if ( !v97 && v101 == -16 )
              v97 = v64;
            v100 = v96 & (unsigned int)(v100 + v99);
            v64 = (unsigned __int64 *)(v98 + 72 * v100);
            v101 = *v64;
            if ( v16 == *v64 )
              goto LABEL_54;
            ++v99;
          }
          if ( v97 )
            v64 = v97;
        }
        goto LABEL_54;
      }
LABEL_158:
      ++*(_DWORD *)(v17 + 320);
      BUG();
    }
LABEL_99:
    sub_209A280(v17 + 304, 2 * v18);
    v88 = *(_DWORD *)(v17 + 328);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = *(_QWORD *)(v17 + 312);
      v91 = v89 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v66 = *(_DWORD *)(v17 + 320) + 1;
      v64 = (unsigned __int64 *)(v90 + 72LL * v91);
      v92 = *v64;
      if ( v16 != *v64 )
      {
        v93 = 1;
        v94 = 0;
        while ( v92 != -8 )
        {
          if ( !v94 && v92 == -16 )
            v94 = v64;
          v91 = v89 & (v93 + v91);
          v64 = (unsigned __int64 *)(v90 + 72LL * v91);
          v92 = *v64;
          if ( v16 == *v64 )
            goto LABEL_54;
          ++v93;
        }
        if ( v94 )
          v64 = v94;
      }
      goto LABEL_54;
    }
    goto LABEL_158;
  }
LABEL_13:
  v23 = *((unsigned int *)v21 + 16);
  v24 = v21[6];
  v25 = v21[2];
  v26 = *((unsigned int *)v21 + 8);
  if ( (_DWORD)v23 )
  {
    v27 = (v23 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v28 = (__int64 **)(v24 + 16LL * v27);
    v29 = *v28;
    if ( v14 == *v28 )
    {
LABEL_15:
      if ( v28 != (__int64 **)(v24 + 16 * v23) )
        v14 = v28[1];
    }
    else
    {
      v86 = 1;
      while ( v29 != (__int64 *)-8LL )
      {
        v87 = v86 + 1;
        v27 = (v23 - 1) & (v86 + v27);
        v28 = (__int64 **)(v24 + 16LL * v27);
        v29 = *v28;
        if ( v14 == *v28 )
          goto LABEL_15;
        v86 = v87;
      }
    }
  }
  if ( (_DWORD)v26 )
  {
    v30 = (v26 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v31 = v25 + 16LL * v30;
    v32 = *(__int64 **)v31;
    if ( v14 == *(__int64 **)v31 )
      goto LABEL_19;
    v84 = 1;
    while ( v32 != (__int64 *)-8LL )
    {
      v85 = v84 + 1;
      v30 = (v26 - 1) & (v84 + v30);
      v31 = v25 + 16LL * v30;
      v32 = *(__int64 **)v31;
      if ( v14 == *(__int64 **)v31 )
        goto LABEL_19;
      v84 = v85;
    }
  }
LABEL_57:
  v31 = v25 + 16 * v26;
LABEL_19:
  v123 = a1 + 8;
  if ( !*(_BYTE *)(v31 + 12) )
  {
    *(_QWORD *)&v131 = a2;
    result = sub_205F5C0(v123, (__int64 *)&v131);
    result[1] = (__int64)v118;
    *((_DWORD *)result + 4) = v115;
    return result;
  }
  v35 = *(_QWORD **)(a1 + 552);
  v36 = *(_DWORD *)(v31 + 8);
  v37 = sub_1E0A0C0(v35[4]);
  v38 = 8 * sub_15A9520(v37, *(_DWORD *)(v37 + 4));
  if ( v38 == 32 )
  {
    v39 = 5;
  }
  else if ( v38 > 0x20 )
  {
    v39 = 6;
    if ( v38 != 64 )
    {
      v39 = 0;
      if ( v38 == 128 )
        v39 = 7;
    }
  }
  else
  {
    v39 = 3;
    if ( v38 != 8 )
      v39 = 4 * (v38 == 16);
  }
  v119 = sub_1D299D0(v35, v36, v39, 0, 1);
  v120 = v40;
  v41 = sub_2051C20((__int64 *)a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  v42 = *(_QWORD **)(a1 + 552);
  v117 = v43;
  v44 = v42[4];
  v116 = (__int64)v41;
  memset(v130, 0, 24);
  sub_1E341E0((__int64)&v131, v44, v36, 0);
  v45 = *(_DWORD *)(a1 + 536);
  v46 = *(_QWORD *)a1;
  v126 = 0;
  v127 = v45;
  if ( v46 )
  {
    if ( &v126 != (__int64 *)(v46 + 48) )
    {
      v47 = *(_QWORD *)(v46 + 48);
      v126 = v47;
      if ( v47 )
        sub_1623A60((__int64)&v126, v47, 2);
    }
  }
  v48 = *(_QWORD *)a2;
  v49 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 552) + 32LL));
  v50 = *(_BYTE *)(v48 + 8);
  if ( v50 == 15 )
  {
    v51 = 8 * sub_15A9520(v49, *(_DWORD *)(v48 + 8) >> 8);
    if ( v51 == 32 )
    {
      LOBYTE(v52) = 5;
    }
    else if ( v51 > 0x20 )
    {
      LOBYTE(v52) = 6;
      if ( v51 != 64 )
      {
        LOBYTE(v52) = 0;
        if ( v51 == 128 )
          LOBYTE(v52) = 7;
      }
    }
    else
    {
      LOBYTE(v52) = 3;
      if ( v51 != 8 )
        LOBYTE(v52) = 4 * (v51 == 16);
    }
    v53 = 0;
  }
  else if ( v50 == 16 )
  {
    v75 = *(_QWORD *)(v48 + 24);
    if ( *(_BYTE *)(v75 + 8) == 15 )
    {
      v76 = 8 * sub_15A9520(v49, *(_DWORD *)(v75 + 8) >> 8);
      if ( v76 == 32 )
      {
        v77 = 5;
      }
      else if ( v76 > 0x20 )
      {
        v77 = 6;
        if ( v76 != 64 )
        {
          v77 = 0;
          if ( v76 == 128 )
            v77 = 7;
        }
      }
      else
      {
        v77 = 3;
        if ( v76 != 8 )
          v77 = 4 * (v76 == 16);
      }
      v128[0] = v77;
      v129 = 0;
      v75 = sub_1F58E60((__int64)v128, *(_QWORD **)v48);
    }
    v122 = *(_QWORD *)(v48 + 32);
    LOBYTE(v78) = sub_1F59570(v75);
    v79 = *(_QWORD **)v48;
    v113 = v80;
    v114 = v78;
    LOBYTE(v52) = sub_1D15020(v78, v122);
    v53 = 0;
    if ( !(_BYTE)v52 )
    {
      v52 = sub_1F593D0(v79, v114, v113, v122);
      v112 = v52;
    }
    v81 = v112;
    LOBYTE(v81) = v52;
    v121 = v81;
  }
  else
  {
    LOBYTE(v52) = sub_1F59570(v48);
    v121 = v52;
  }
  v54 = v121;
  LOBYTE(v54) = v52;
  v55 = sub_1D2B730(v42, v54, v53, (__int64)&v126, v116, v117, (__int64)v119, v120, v131, v132, 0, 0, (__int64)v130, 0);
  v57 = v56;
  if ( v126 )
    sub_161E7C0((__int64)&v126, v126);
  v58 = *(_QWORD *)(a1 + 552);
  if ( v55 )
  {
    nullsub_686();
    *(_QWORD *)(v58 + 176) = v55;
    *(_DWORD *)(v58 + 184) = 1;
    sub_1D23870();
    v59 = *(_DWORD *)(a1 + 32);
    if ( v59 )
      goto LABEL_42;
LABEL_63:
    ++*(_QWORD *)(a1 + 8);
    goto LABEL_64;
  }
  *(_QWORD *)(v58 + 176) = 0;
  *(_DWORD *)(v58 + 184) = 1;
  v59 = *(_DWORD *)(a1 + 32);
  if ( !v59 )
    goto LABEL_63;
LABEL_42:
  v60 = *(_QWORD *)(a1 + 16);
  v61 = (v59 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v62 = (__int64 *)(v60 + 24LL * v61);
  v63 = *v62;
  if ( a2 != *v62 )
  {
    v102 = 1;
    v103 = 0;
    while ( v63 != -8 )
    {
      if ( v63 == -16 && !v103 )
        v103 = v62;
      v61 = (v59 - 1) & (v102 + v61);
      v62 = (__int64 *)(v60 + 24LL * v61);
      v63 = *v62;
      if ( a2 == *v62 )
        goto LABEL_43;
      ++v102;
    }
    v104 = *(_DWORD *)(a1 + 24);
    if ( v103 )
      v62 = v103;
    ++*(_QWORD *)(a1 + 8);
    v71 = v104 + 1;
    if ( 4 * (v104 + 1) < 3 * v59 )
    {
      if ( v59 - *(_DWORD *)(a1 + 28) - v71 > v59 >> 3 )
      {
LABEL_118:
        *(_DWORD *)(a1 + 24) = v71;
        if ( *v62 != -8 )
          --*(_DWORD *)(a1 + 28);
        *v62 = a2;
        v62[1] = 0;
        *((_DWORD *)v62 + 4) = 0;
        goto LABEL_43;
      }
      sub_205F3F0(v123, v59);
      v105 = *(_DWORD *)(a1 + 32);
      if ( v105 )
      {
        v106 = v105 - 1;
        v107 = *(_QWORD *)(a1 + 16);
        v108 = 1;
        LODWORD(v109) = (v105 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v110 = 0;
        v71 = *(_DWORD *)(a1 + 24) + 1;
        v62 = (__int64 *)(v107 + 24LL * (unsigned int)v109);
        v111 = *v62;
        if ( a2 != *v62 )
        {
          while ( v111 != -8 )
          {
            if ( v111 == -16 && !v110 )
              v110 = v62;
            v109 = v106 & (unsigned int)(v109 + v108);
            v62 = (__int64 *)(v107 + 24 * v109);
            v111 = *v62;
            if ( a2 == *v62 )
              goto LABEL_118;
            ++v108;
          }
          if ( v110 )
            v62 = v110;
        }
        goto LABEL_118;
      }
LABEL_159:
      ++*(_DWORD *)(a1 + 24);
      BUG();
    }
LABEL_64:
    sub_205F3F0(v123, 2 * v59);
    v67 = *(_DWORD *)(a1 + 32);
    if ( v67 )
    {
      v68 = v67 - 1;
      v69 = *(_QWORD *)(a1 + 16);
      v70 = (v67 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v71 = *(_DWORD *)(a1 + 24) + 1;
      v62 = (__int64 *)(v69 + 24LL * v70);
      v72 = *v62;
      if ( a2 != *v62 )
      {
        v73 = 1;
        v74 = 0;
        while ( v72 != -8 )
        {
          if ( !v74 && v72 == -16 )
            v74 = v62;
          v70 = v68 & (v73 + v70);
          v62 = (__int64 *)(v69 + 24LL * v70);
          v72 = *v62;
          if ( a2 == *v62 )
            goto LABEL_118;
          ++v73;
        }
        if ( v74 )
          v62 = v74;
      }
      goto LABEL_118;
    }
    goto LABEL_159;
  }
LABEL_43:
  v62[1] = v55;
  *((_DWORD *)v62 + 4) = v57;
  return (__int64 *)v57;
}
