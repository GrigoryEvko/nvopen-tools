// Function: sub_1BE6920
// Address: 0x1be6920
//
__int64 __fastcall sub_1BE6920(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  _QWORD *v11; // rbx
  __int64 *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  __int64 v15; // rdi
  int v16; // ecx
  int v17; // ecx
  __int64 v18; // rsi
  unsigned int v19; // edx
  _QWORD *v20; // rax
  _QWORD *v21; // r10
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 *v24; // r14
  _QWORD *v25; // rax
  _QWORD *v26; // r12
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int8 *v33; // rsi
  __int64 *v34; // r14
  __int64 v35; // rsi
  __int64 v36; // rsi
  unsigned __int8 *v37; // rsi
  _BYTE *v38; // rsi
  _QWORD *v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  unsigned __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rdx
  __int64 v47; // rax
  char v48; // si
  __int64 v49; // rcx
  __int64 v50; // r8
  unsigned __int64 v51; // r13
  __int64 v52; // rax
  unsigned __int64 v53; // rdi
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  char v56; // si
  unsigned __int64 v57; // rax
  __int64 v58; // rcx
  unsigned __int64 v59; // rax
  char v60; // si
  char v61; // r8
  __int64 v62; // r13
  _QWORD *v63; // rax
  _QWORD *v64; // rdi
  double v65; // xmm4_8
  double v66; // xmm5_8
  __int64 *v68; // rax
  __int64 *v69; // r12
  __int64 *v70; // rbx
  unsigned int v71; // esi
  __int64 v72; // rcx
  __int64 v73; // r8
  unsigned int v74; // edx
  _QWORD *v75; // rax
  __int64 v76; // rdi
  __int64 v77; // rdx
  int v78; // ecx
  __int64 v79; // rdi
  int v80; // ecx
  __int64 v81; // r14
  int v82; // edx
  unsigned int v83; // esi
  __int64 v84; // r8
  int v85; // r11d
  _QWORD *v86; // r10
  int v87; // r14d
  _QWORD *v88; // r11
  int v89; // ecx
  __int64 v90; // rdx
  int v91; // esi
  __int64 v92; // rdi
  int v93; // esi
  int v94; // r11d
  __int64 v95; // r14
  unsigned int v96; // ecx
  __int64 v97; // r8
  int v98; // eax
  int v99; // edi
  __int64 *v100; // [rsp+8h] [rbp-258h]
  __int64 v101; // [rsp+10h] [rbp-250h]
  __int64 v103; // [rsp+20h] [rbp-240h]
  __int64 v104; // [rsp+28h] [rbp-238h]
  _QWORD v105[2]; // [rsp+30h] [rbp-230h] BYREF
  unsigned __int64 v106; // [rsp+40h] [rbp-220h]
  _BYTE v107[64]; // [rsp+58h] [rbp-208h] BYREF
  __int64 v108; // [rsp+98h] [rbp-1C8h]
  __int64 v109; // [rsp+A0h] [rbp-1C0h]
  unsigned __int64 v110; // [rsp+A8h] [rbp-1B8h]
  _QWORD v111[2]; // [rsp+B0h] [rbp-1B0h] BYREF
  unsigned __int64 v112; // [rsp+C0h] [rbp-1A0h]
  _BYTE v113[64]; // [rsp+D8h] [rbp-188h] BYREF
  unsigned __int64 v114; // [rsp+118h] [rbp-148h]
  unsigned __int64 i; // [rsp+120h] [rbp-140h]
  unsigned __int64 v116; // [rsp+128h] [rbp-138h]
  __int64 v117[2]; // [rsp+130h] [rbp-130h] BYREF
  unsigned __int64 v118; // [rsp+140h] [rbp-120h]
  __int64 v119; // [rsp+198h] [rbp-C8h]
  __int64 v120; // [rsp+1A0h] [rbp-C0h]
  __int64 v121; // [rsp+1A8h] [rbp-B8h]
  char v122[8]; // [rsp+1B0h] [rbp-B0h] BYREF
  __int64 v123; // [rsp+1B8h] [rbp-A8h]
  unsigned __int64 v124; // [rsp+1C0h] [rbp-A0h]
  __int64 v125; // [rsp+218h] [rbp-48h]
  __int64 v126; // [rsp+220h] [rbp-40h]
  __int64 v127; // [rsp+228h] [rbp-38h]

  if ( *(_DWORD *)(a1 + 296) )
  {
    v68 = *(__int64 **)(a1 + 288);
    v69 = &v68[2 * *(unsigned int *)(a1 + 304)];
    if ( v68 != v69 )
    {
      while ( 1 )
      {
        v70 = v68;
        if ( *v68 != -16 && *v68 != -8 )
          break;
        v68 += 2;
        if ( v69 == v68 )
          goto LABEL_2;
      }
      if ( v69 != v68 )
      {
        v71 = *(_DWORD *)(a2 + 216);
        if ( !v71 )
          goto LABEL_92;
        while ( 1 )
        {
          v72 = v70[1];
          v73 = *(_QWORD *)(a2 + 200);
          v74 = (v71 - 1) & (((unsigned int)v72 >> 9) ^ ((unsigned int)v72 >> 4));
          v75 = (_QWORD *)(v73 + 16LL * v74);
          v76 = *v75;
          if ( v72 != *v75 )
          {
            v87 = 1;
            v88 = 0;
            while ( v76 != -8 )
            {
              if ( !v88 && v76 == -16 )
                v88 = v75;
              v74 = (v71 - 1) & (v87 + v74);
              v75 = (_QWORD *)(v73 + 16LL * v74);
              v76 = *v75;
              if ( v72 == *v75 )
                goto LABEL_85;
              ++v87;
            }
            v89 = *(_DWORD *)(a2 + 208);
            if ( v88 )
              v75 = v88;
            ++*(_QWORD *)(a2 + 192);
            v82 = v89 + 1;
            if ( 4 * (v89 + 1) >= 3 * v71 )
              goto LABEL_93;
            if ( v71 - *(_DWORD *)(a2 + 212) - v82 <= v71 >> 3 )
            {
              sub_1BA1560(a2 + 192, v71);
              v91 = *(_DWORD *)(a2 + 216);
              if ( !v91 )
              {
LABEL_134:
                ++*(_DWORD *)(a2 + 208);
                BUG();
              }
              v92 = v70[1];
              v93 = v91 - 1;
              v94 = 1;
              v86 = 0;
              v95 = *(_QWORD *)(a2 + 200);
              v82 = *(_DWORD *)(a2 + 208) + 1;
              v96 = v93 & (((unsigned int)v92 >> 9) ^ ((unsigned int)v92 >> 4));
              v75 = (_QWORD *)(v95 + 16LL * v96);
              v97 = *v75;
              if ( v92 != *v75 )
              {
                while ( v97 != -8 )
                {
                  if ( !v86 && v97 == -16 )
                    v86 = v75;
                  v96 = v93 & (v94 + v96);
                  v75 = (_QWORD *)(v95 + 16LL * v96);
                  v97 = *v75;
                  if ( v92 == *v75 )
                    goto LABEL_106;
                  ++v94;
                }
                goto LABEL_97;
              }
            }
LABEL_106:
            *(_DWORD *)(a2 + 208) = v82;
            if ( *v75 != -8 )
              --*(_DWORD *)(a2 + 212);
            v90 = v70[1];
            v75[1] = 0;
            *v75 = v90;
          }
LABEL_85:
          v77 = *v70;
          v70 += 2;
          v75[1] = v77;
          if ( v70 == v69 )
            break;
          while ( *v70 == -16 || *v70 == -8 )
          {
            v70 += 2;
            if ( v69 == v70 )
              goto LABEL_2;
          }
          if ( v69 == v70 )
            break;
          v71 = *(_DWORD *)(a2 + 216);
          if ( !v71 )
          {
LABEL_92:
            ++*(_QWORD *)(a2 + 192);
LABEL_93:
            sub_1BA1560(a2 + 192, 2 * v71);
            v78 = *(_DWORD *)(a2 + 216);
            if ( !v78 )
              goto LABEL_134;
            v79 = v70[1];
            v80 = v78 - 1;
            v81 = *(_QWORD *)(a2 + 200);
            v82 = *(_DWORD *)(a2 + 208) + 1;
            v83 = v80 & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
            v75 = (_QWORD *)(v81 + 16LL * v83);
            v84 = *v75;
            if ( v79 != *v75 )
            {
              v85 = 1;
              v86 = 0;
              while ( v84 != -8 )
              {
                if ( v84 == -16 && !v86 )
                  v86 = v75;
                v83 = v80 & (v85 + v83);
                v75 = (_QWORD *)(v81 + 16LL * v83);
                v84 = *v75;
                if ( v79 == *v75 )
                  goto LABEL_106;
                ++v85;
              }
LABEL_97:
              if ( v86 )
                v75 = v86;
              goto LABEL_106;
            }
            goto LABEL_106;
          }
        }
      }
    }
  }
LABEL_2:
  v101 = *(_QWORD *)(a2 + 64);
  v11 = (_QWORD *)sub_157F1C0(v101);
  LOWORD(v118) = 259;
  v117[0] = (__int64)"vector.body.latch";
  v12 = (__int64 *)sub_157EE30((__int64)v11);
  v13 = sub_157FBF0(v11, v12, (__int64)v117);
  v14 = *(_QWORD *)(a2 + 160);
  v15 = 0;
  v104 = v13;
  v16 = *(_DWORD *)(v14 + 24);
  if ( v16 )
  {
    v17 = v16 - 1;
    v18 = *(_QWORD *)(v14 + 8);
    v19 = v17 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
    v20 = (_QWORD *)(v18 + 16LL * v19);
    v21 = (_QWORD *)*v20;
    if ( v11 == (_QWORD *)*v20 )
    {
LABEL_4:
      v15 = v20[1];
    }
    else
    {
      v98 = 1;
      while ( v21 != (_QWORD *)-8LL )
      {
        v99 = v98 + 1;
        v19 = v17 & (v98 + v19);
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = (_QWORD *)*v20;
        if ( v11 == (_QWORD *)*v20 )
          goto LABEL_4;
        v98 = v99;
      }
      v15 = 0;
    }
  }
  sub_1400330(v15, v104, v14);
  v22 = (_QWORD *)sub_157EBA0((__int64)v11);
  sub_15F20C0(v22);
  v23 = *(_QWORD *)(a2 + 176);
  *(_QWORD *)(v23 + 8) = v11;
  *(_QWORD *)(v23 + 16) = v11 + 5;
  v24 = *(__int64 **)(a2 + 176);
  LOWORD(v118) = 257;
  v25 = sub_1648A60(56, 0);
  v26 = v25;
  if ( v25 )
    sub_15F82A0((__int64)v25, v24[3], 0);
  v27 = v24[1];
  v103 = (__int64)(v26 + 3);
  if ( v27 )
  {
    v100 = (__int64 *)v24[2];
    sub_157E9D0(v27 + 40, (__int64)v26);
    v28 = *v100;
    v29 = v26[3] & 7LL;
    v26[4] = v100;
    v28 &= 0xFFFFFFFFFFFFFFF8LL;
    v26[3] = v28 | v29;
    *(_QWORD *)(v28 + 8) = v103;
    *v100 = v103 | *v100 & 7;
  }
  sub_164B780((__int64)v26, v117);
  v30 = *v24;
  if ( *v24 )
  {
    v111[0] = *v24;
    sub_1623A60((__int64)v111, v30, 2);
    v31 = v26[6];
    v32 = (__int64)(v26 + 6);
    if ( v31 )
    {
      sub_161E7C0((__int64)(v26 + 6), v31);
      v32 = (__int64)(v26 + 6);
    }
    v33 = (unsigned __int8 *)v111[0];
    v26[6] = v111[0];
    if ( v33 )
      sub_1623210((__int64)v111, v33, v32);
  }
  v34 = *(__int64 **)(a2 + 176);
  v34[1] = v26[5];
  v34[2] = v103;
  v35 = v26[6];
  v117[0] = v35;
  if ( v35 )
  {
    sub_1623A60((__int64)v117, v35, 2);
    v36 = *v34;
    if ( !*v34 )
      goto LABEL_17;
  }
  else
  {
    v36 = *v34;
    if ( !*v34 )
      goto LABEL_19;
  }
  sub_161E7C0((__int64)v34, v36);
LABEL_17:
  v37 = (unsigned __int8 *)v117[0];
  *v34 = v117[0];
  if ( v37 )
  {
    sub_1623210((__int64)v117, v37, (__int64)v34);
  }
  else if ( v117[0] )
  {
    sub_161E7C0((__int64)v117, v117[0]);
  }
LABEL_19:
  *(_QWORD *)(a2 + 64) = v11;
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a2 + 72) = v104;
  sub_1BE3E20(v117, *(_QWORD *)a1);
  v38 = v107;
  v39 = v105;
  sub_16CCCB0(v105, (__int64)v107, (__int64)v117);
  v41 = v120;
  v42 = v119;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  v43 = v120 - v119;
  if ( v120 == v119 )
  {
    v43 = 0;
    v45 = 0;
  }
  else
  {
    if ( v43 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_128;
    v44 = sub_22077B0(v120 - v119);
    v41 = v120;
    v42 = v119;
    v45 = v44;
  }
  v108 = v45;
  v109 = v45;
  v110 = v45 + v43;
  if ( v41 != v42 )
  {
    v46 = v45;
    v47 = v42;
    do
    {
      if ( v46 )
      {
        *(_QWORD *)v46 = *(_QWORD *)v47;
        v48 = *(_BYTE *)(v47 + 16);
        *(_BYTE *)(v46 + 16) = v48;
        if ( v48 )
          *(_QWORD *)(v46 + 8) = *(_QWORD *)(v47 + 8);
      }
      v47 += 24;
      v46 += 24;
    }
    while ( v41 != v47 );
    v45 += 8 * ((unsigned __int64)(v41 - 24 - v42) >> 3) + 24;
  }
  v109 = v45;
  v39 = v111;
  v38 = v113;
  sub_16CCCB0(v111, (__int64)v113, (__int64)v122);
  v49 = v126;
  v50 = v125;
  v114 = 0;
  i = 0;
  v116 = 0;
  v51 = v126 - v125;
  if ( v126 == v125 )
  {
    v53 = 0;
    goto LABEL_32;
  }
  if ( v51 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_128:
    sub_4261EA(v39, v38, v40);
  v52 = sub_22077B0(v126 - v125);
  v49 = v126;
  v50 = v125;
  v53 = v52;
LABEL_32:
  v114 = v53;
  i = v53;
  v116 = v53 + v51;
  if ( v49 == v50 )
  {
    v57 = v53;
  }
  else
  {
    v54 = v53;
    v55 = v50;
    do
    {
      if ( v54 )
      {
        *(_QWORD *)v54 = *(_QWORD *)v55;
        v56 = *(_BYTE *)(v55 + 16);
        *(_BYTE *)(v54 + 16) = v56;
        if ( v56 )
          *(_QWORD *)(v54 + 8) = *(_QWORD *)(v55 + 8);
      }
      v55 += 24;
      v54 += 24LL;
    }
    while ( v49 != v55 );
    v57 = v53 + 8 * ((unsigned __int64)(v49 - 24 - v50) >> 3) + 24;
  }
  for ( i = v57; ; v57 = i )
  {
    v58 = v108;
    if ( v109 - v108 != v57 - v53 )
      goto LABEL_40;
    if ( v108 == v109 )
      break;
    v59 = v53;
    while ( *(_QWORD *)v58 == *(_QWORD *)v59 )
    {
      v60 = *(_BYTE *)(v58 + 16);
      v61 = *(_BYTE *)(v59 + 16);
      if ( v60 && v61 )
      {
        if ( *(_QWORD *)(v58 + 8) != *(_QWORD *)(v59 + 8) )
          break;
        v58 += 24;
        v59 += 24LL;
        if ( v109 == v58 )
          goto LABEL_49;
      }
      else
      {
        if ( v61 != v60 )
          break;
        v58 += 24;
        v59 += 24LL;
        if ( v109 == v58 )
          goto LABEL_49;
      }
    }
LABEL_40:
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(v109 - 24) + 16LL))(*(_QWORD *)(v109 - 24), a2);
    sub_1BE4140((__int64)v105);
    v53 = v114;
  }
LABEL_49:
  if ( v53 )
    j_j___libc_free_0(v53, v116 - v53);
  if ( v112 != v111[1] )
    _libc_free(v112);
  if ( v108 )
    j_j___libc_free_0(v108, v110 - v108);
  if ( v106 != v105[1] )
    _libc_free(v106);
  if ( v125 )
    j_j___libc_free_0(v125, v127 - v125);
  if ( v124 != v123 )
    _libc_free(v124);
  if ( v119 )
    j_j___libc_free_0(v119, v121 - v119);
  if ( v118 != v117[1] )
    _libc_free(v118);
  v62 = *(_QWORD *)(a2 + 64);
  v63 = (_QWORD *)sub_157EBA0(v62);
  sub_15F20C0(v63);
  v64 = sub_1648A60(56, 1u);
  if ( v64 )
    sub_15F8590((__int64)v64, v104, v62);
  sub_1AA7EA0(v104, 0, *(_QWORD *)(a2 + 160), 0, 0, a3, a4, a5, a6, v65, v66, a9, a10);
  return sub_1BE6790(*(_QWORD *)(a2 + 168), v101, v62);
}
