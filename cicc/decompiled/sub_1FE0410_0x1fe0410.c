// Function: sub_1FE0410
// Address: 0x1fe0410
//
void __fastcall sub_1FE0410(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v4; // rdi
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // rax
  unsigned int v11; // esi
  __int64 v12; // r8
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned int v19; // ecx
  __int64 v20; // rsi
  __int64 v21; // r14
  __int64 *v22; // rax
  __int64 v23; // rsi
  char v24; // al
  __int64 v25; // rsi
  int v26; // eax
  unsigned __int64 v27; // rax
  unsigned int v28; // esi
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // rdi
  char v32; // cl
  unsigned int v33; // r13d
  __int64 v34; // rdx
  __int64 v35; // rax
  char v36; // dl
  unsigned int v37; // eax
  __int64 v38; // rdx
  unsigned __int64 v39; // rax
  unsigned int v40; // eax
  unsigned __int64 v41; // rsi
  unsigned __int64 v42; // rdx
  unsigned int v43; // esi
  __int64 v44; // r10
  unsigned int v45; // edi
  __int64 *v46; // rdx
  __int64 v47; // rcx
  int v48; // esi
  __int64 v49; // rax
  __int64 v50; // r15
  unsigned int v51; // edx
  bool v52; // cc
  unsigned __int64 v53; // rax
  __int64 v54; // rdi
  __int64 v55; // rdi
  unsigned int v56; // eax
  __int64 v57; // r14
  __int64 v58; // r14
  __int64 v59; // rbx
  unsigned int v60; // r15d
  unsigned int v61; // ecx
  unsigned int v62; // ecx
  int v63; // esi
  int v64; // r11d
  __int64 *v65; // rdx
  int v66; // eax
  int v67; // ecx
  int v68; // r15d
  __int64 *v69; // r9
  __int64 *v70; // rcx
  int v71; // ebx
  int v72; // edx
  __int64 v73; // rax
  __int64 v74; // r15
  unsigned __int64 *v75; // rdi
  int v76; // eax
  int v77; // esi
  __int64 v78; // rdi
  unsigned int v79; // eax
  __int64 v80; // r8
  int v81; // r11d
  __int64 *v82; // r9
  int v83; // edx
  int v84; // r9d
  __int64 v85; // rdi
  __int64 v86; // rsi
  __int64 v87; // r10
  int v88; // r8d
  __int64 *v89; // r11
  int v90; // eax
  int v91; // eax
  __int64 v92; // rdi
  __int64 *v93; // r8
  unsigned int v94; // r14d
  int v95; // r10d
  __int64 v96; // rsi
  __int64 v97; // r14
  __int64 v98; // rbx
  unsigned int v99; // r15d
  __int64 v100; // rdi
  __int64 v101; // rdi
  int v102; // edx
  int v103; // esi
  __int64 v104; // r9
  int v105; // r8d
  __int64 v106; // rdi
  __int64 v107; // r10
  int v108; // ecx
  unsigned __int64 v109; // rax
  unsigned __int64 v110; // rdx
  __int64 *v111; // [rsp+10h] [rbp-F0h]
  unsigned __int64 *v112; // [rsp+18h] [rbp-E8h]
  __int64 v113; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v114; // [rsp+20h] [rbp-E0h]
  unsigned int v115; // [rsp+20h] [rbp-E0h]
  __int64 v116; // [rsp+20h] [rbp-E0h]
  unsigned int v117; // [rsp+20h] [rbp-E0h]
  __int64 v118; // [rsp+28h] [rbp-D8h]
  int v119; // [rsp+38h] [rbp-C8h]
  unsigned int v120; // [rsp+38h] [rbp-C8h]
  unsigned int v121; // [rsp+38h] [rbp-C8h]
  unsigned int v122; // [rsp+38h] [rbp-C8h]
  unsigned int v123; // [rsp+48h] [rbp-B8h]
  __int64 v124; // [rsp+48h] [rbp-B8h]
  __int64 v125; // [rsp+48h] [rbp-B8h]
  __int64 v126; // [rsp+58h] [rbp-A8h] BYREF
  __m128i v127; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v128; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v129; // [rsp+78h] [rbp-88h]
  unsigned __int64 v130; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v131; // [rsp+88h] [rbp-78h]
  const __m128i *v132[2]; // [rsp+90h] [rbp-70h] BYREF
  _BYTE v133[16]; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v134; // [rsp+B0h] [rbp-50h] BYREF
  unsigned int v135; // [rsp+B8h] [rbp-48h]
  __int64 v136; // [rsp+C0h] [rbp-40h] BYREF
  unsigned int v137; // [rsp+C8h] [rbp-38h]

  v2 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) != 11 )
    return;
  v4 = *(_QWORD *)(a1 + 8);
  v5 = a2;
  v132[0] = (const __m128i *)v133;
  v132[1] = (const __m128i *)0x100000000LL;
  v6 = sub_1E0A0C0(v4);
  sub_20C7CE0(*(_QWORD *)(a1 + 16), v6, v2, v132, 0, 0);
  v7 = *(_QWORD *)(a1 + 16);
  v127 = _mm_loadu_si128(v132[0]);
  v8 = sub_16498A0(a2);
  if ( (unsigned int)sub_1FDDD20(v7, v8, v127.m128i_u32[0], v127.m128i_i64[1]) != 1 )
    goto LABEL_4;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = sub_16498A0(a2);
  sub_1F40D10((__int64)&v134, v9, v10, v127.m128i_i64[0], v127.m128i_i64[1]);
  v127.m128i_i8[0] = v135;
  v127.m128i_i64[1] = v136;
  if ( (_BYTE)v135 )
    v123 = sub_1FDDC20(v135);
  else
    v123 = sub_1F58D40((__int64)&v127);
  v11 = *(_DWORD *)(a1 + 232);
  v118 = a1 + 208;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 208);
    goto LABEL_148;
  }
  v12 = *(_QWORD *)(a1 + 216);
  v13 = (v11 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v14 = (__int64 *)(v12 + 16LL * v13);
  v15 = *v14;
  if ( v5 != *v14 )
  {
    v64 = 1;
    v65 = 0;
    while ( v15 != -8 )
    {
      if ( !v65 && v15 == -16 )
        v65 = v14;
      v13 = (v11 - 1) & (v64 + v13);
      v14 = (__int64 *)(v12 + 16LL * v13);
      v15 = *v14;
      if ( v5 == *v14 )
        goto LABEL_10;
      ++v64;
    }
    if ( !v65 )
      v65 = v14;
    v66 = *(_DWORD *)(a1 + 224);
    ++*(_QWORD *)(a1 + 208);
    v67 = v66 + 1;
    if ( 4 * (v66 + 1) < 3 * v11 )
    {
      if ( v11 - *(_DWORD *)(a1 + 228) - v67 > v11 >> 3 )
        goto LABEL_127;
      sub_1542080(v118, v11);
      v90 = *(_DWORD *)(a1 + 232);
      if ( v90 )
      {
        v91 = v90 - 1;
        v92 = *(_QWORD *)(a1 + 216);
        v93 = 0;
        v94 = v91 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v95 = 1;
        v67 = *(_DWORD *)(a1 + 224) + 1;
        v65 = (__int64 *)(v92 + 16LL * v94);
        v96 = *v65;
        if ( v5 != *v65 )
        {
          while ( v96 != -8 )
          {
            if ( !v93 && v96 == -16 )
              v93 = v65;
            v94 = v91 & (v95 + v94);
            v65 = (__int64 *)(v92 + 16LL * v94);
            v96 = *v65;
            if ( v5 == *v65 )
              goto LABEL_127;
            ++v95;
          }
          if ( v93 )
            v65 = v93;
        }
        goto LABEL_127;
      }
      goto LABEL_223;
    }
LABEL_148:
    sub_1542080(v118, 2 * v11);
    v76 = *(_DWORD *)(a1 + 232);
    if ( v76 )
    {
      v77 = v76 - 1;
      v78 = *(_QWORD *)(a1 + 216);
      v79 = (v76 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v67 = *(_DWORD *)(a1 + 224) + 1;
      v65 = (__int64 *)(v78 + 16LL * v79);
      v80 = *v65;
      if ( *v65 != v5 )
      {
        v81 = 1;
        v82 = 0;
        while ( v80 != -8 )
        {
          if ( !v82 && v80 == -16 )
            v82 = v65;
          v79 = v77 & (v81 + v79);
          v65 = (__int64 *)(v78 + 16LL * v79);
          v80 = *v65;
          if ( v5 == *v65 )
            goto LABEL_127;
          ++v81;
        }
        if ( v82 )
          v65 = v82;
      }
LABEL_127:
      *(_DWORD *)(a1 + 224) = v67;
      if ( *v65 != -8 )
        --*(_DWORD *)(a1 + 228);
      *v65 = v5;
      *((_DWORD *)v65 + 2) = 0;
      goto LABEL_4;
    }
LABEL_223:
    ++*(_DWORD *)(a1 + 224);
    BUG();
  }
LABEL_10:
  v16 = *((_DWORD *)v14 + 2);
  if ( v16 >= 0 )
    goto LABEL_4;
  v17 = v16 & 0x7FFFFFFF;
  v18 = *(unsigned int *)(a1 + 952);
  v19 = v17 + 1;
  if ( (int)v17 + 1 <= (unsigned int)v18 )
    goto LABEL_12;
  v57 = v19;
  if ( v19 < v18 )
  {
    v20 = *(_QWORD *)(a1 + 944);
    v97 = v20 + 40LL * v19;
    if ( v20 + 40 * v18 != v97 )
    {
      v120 = v17;
      v116 = v5;
      v98 = v20 + 40 * v18;
      v99 = v17 + 1;
      do
      {
        v98 -= 40;
        if ( *(_DWORD *)(v98 + 32) > 0x40u )
        {
          v100 = *(_QWORD *)(v98 + 24);
          if ( v100 )
            j_j___libc_free_0_0(v100);
        }
        if ( *(_DWORD *)(v98 + 16) > 0x40u )
        {
          v101 = *(_QWORD *)(v98 + 8);
          if ( v101 )
            j_j___libc_free_0_0(v101);
        }
      }
      while ( v97 != v98 );
      v17 = v120;
      v5 = v116;
      v19 = v99;
      v20 = *(_QWORD *)(a1 + 944);
    }
  }
  else
  {
    if ( v19 <= v18 )
    {
LABEL_12:
      v20 = *(_QWORD *)(a1 + 944);
      goto LABEL_13;
    }
    if ( v19 > (unsigned __int64)*(unsigned int *)(a1 + 956) )
    {
      v117 = v17 + 1;
      v122 = v17;
      sub_1D4FA80(a1 + 944, v19);
      v18 = *(unsigned int *)(a1 + 952);
      v19 = v117;
      v17 = v122;
    }
    v20 = *(_QWORD *)(a1 + 944);
    v58 = v20 + 40 * v57;
    if ( v58 != v20 + 40 * v18 )
    {
      v113 = v5;
      v59 = v20 + 40 * v18;
      v60 = v17;
      v115 = v19;
      do
      {
        if ( v59 )
        {
          *(_DWORD *)v59 = *(_DWORD *)(a1 + 960);
          v62 = *(_DWORD *)(a1 + 976);
          *(_DWORD *)(v59 + 16) = v62;
          if ( v62 <= 0x40 )
            *(_QWORD *)(v59 + 8) = *(_QWORD *)(a1 + 968);
          else
            sub_16A4FD0(v59 + 8, (const void **)(a1 + 968));
          v61 = *(_DWORD *)(a1 + 992);
          *(_DWORD *)(v59 + 32) = v61;
          if ( v61 > 0x40 )
            sub_16A4FD0(v59 + 24, (const void **)(a1 + 984));
          else
            *(_QWORD *)(v59 + 24) = *(_QWORD *)(a1 + 984);
        }
        v59 += 40;
      }
      while ( v58 != v59 );
      v19 = v115;
      v5 = v113;
      v17 = v60;
      v20 = *(_QWORD *)(a1 + 944);
    }
  }
  *(_DWORD *)(a1 + 952) = v19;
LABEL_13:
  v21 = v20 + 40 * v17;
  if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
    v22 = *(__int64 **)(v5 - 8);
  else
    v22 = (__int64 *)(v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF));
  v23 = *v22;
  v24 = *(_BYTE *)(*v22 + 16);
  v126 = v23;
  if ( ((v24 - 5) & 0xFB) == 0 )
  {
LABEL_87:
    *(_DWORD *)v21 = *(_DWORD *)v21 & 0x80000000 | 1;
    v135 = v123;
    if ( v123 > 0x40 )
    {
      sub_16A4EF0((__int64)&v134, 0, 0);
      v137 = v123;
      sub_16A4EF0((__int64)&v136, 0, 0);
    }
    else
    {
      v134 = 0;
      v137 = v123;
      v136 = 0;
    }
    if ( *(_DWORD *)(v21 + 16) > 0x40u )
    {
      v54 = *(_QWORD *)(v21 + 8);
      if ( v54 )
        j_j___libc_free_0_0(v54);
    }
    *(_QWORD *)(v21 + 8) = v134;
    *(_DWORD *)(v21 + 16) = v135;
    v135 = 0;
    if ( *(_DWORD *)(v21 + 32) > 0x40u && (v55 = *(_QWORD *)(v21 + 24)) != 0 )
    {
      j_j___libc_free_0_0(v55);
      v56 = v135;
      *(_QWORD *)(v21 + 24) = v136;
      *(_DWORD *)(v21 + 32) = v137;
      if ( v56 > 0x40 && v134 )
        j_j___libc_free_0_0(v134);
    }
    else
    {
      *(_QWORD *)(v21 + 24) = v136;
      *(_DWORD *)(v21 + 32) = v137;
    }
    goto LABEL_4;
  }
  if ( v24 != 13 )
  {
    v63 = *((_DWORD *)sub_1FDFF40(v118, &v126) + 2);
    if ( v63 >= 0 )
      goto LABEL_113;
    v73 = sub_1FDE940(a1, v63, v123);
    v74 = v73;
    if ( !v73 )
      goto LABEL_113;
    v75 = (unsigned __int64 *)(v21 + 8);
    *(_DWORD *)v21 = *(_DWORD *)v73 & 0x7FFFFFFF | *(_DWORD *)v21 & 0x80000000;
    v52 = *(_DWORD *)(v21 + 16) <= 0x40u;
    *(_BYTE *)(v21 + 3) = *(_BYTE *)(v73 + 3) & 0x80 | *(_BYTE *)(v21 + 3) & 0x7F;
    if ( v52 && *(_DWORD *)(v73 + 16) <= 0x40u )
    {
      *(_QWORD *)(v21 + 8) = *(_QWORD *)(v73 + 8);
      *(_DWORD *)(v21 + 16) = *(_DWORD *)(v73 + 16);
      sub_1FDDCF0(v75);
    }
    else
    {
      sub_16A51C0((__int64)v75, v73 + 8);
    }
    v112 = (unsigned __int64 *)(v21 + 24);
    if ( *(_DWORD *)(v21 + 32) <= 0x40u && *(_DWORD *)(v74 + 32) <= 0x40u )
    {
      *(_QWORD *)(v21 + 24) = *(_QWORD *)(v74 + 24);
      *(_DWORD *)(v21 + 32) = *(_DWORD *)(v74 + 32);
      sub_1FDDCF0(v112);
    }
    else
    {
      sub_16A51C0((__int64)v112, v74 + 24);
    }
    goto LABEL_35;
  }
  sub_16A5D10((__int64)&v130, v23 + 24, v123);
  v25 = 1LL << ((unsigned __int8)v131 - 1);
  if ( v131 > 0x40 )
  {
    if ( (*(_QWORD *)(v130 + 8LL * ((v131 - 1) >> 6)) & v25) != 0 )
      v26 = sub_16A5810((__int64)&v130);
    else
      v26 = sub_16A57B0((__int64)&v130);
  }
  else if ( (v130 & v25) != 0 )
  {
    v26 = 64;
    if ( v130 << (64 - (unsigned __int8)v131) != -1 )
    {
      _BitScanReverse64(&v27, ~(v130 << (64 - (unsigned __int8)v131)));
      v26 = v27 ^ 0x3F;
    }
  }
  else
  {
    v108 = 64;
    if ( v130 )
    {
      _BitScanReverse64(&v109, v130);
      v108 = v109 ^ 0x3F;
    }
    v26 = v131 + v108 - 64;
  }
  *(_DWORD *)v21 = v26 & 0x7FFFFFFF | *(_DWORD *)v21 & 0x80000000;
  v28 = v131;
  v135 = v131;
  if ( v131 <= 0x40 )
  {
    v29 = v130;
LABEL_23:
    v30 = ~v29 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v28);
    v134 = v30;
    goto LABEL_24;
  }
  sub_16A4FD0((__int64)&v134, (const void **)&v130);
  v28 = v135;
  if ( v135 <= 0x40 )
  {
    v29 = v134;
    goto LABEL_23;
  }
  sub_16A8F40((__int64 *)&v134);
  v28 = v135;
  v30 = v134;
LABEL_24:
  v135 = 0;
  if ( *(_DWORD *)(v21 + 16) > 0x40u )
  {
    v31 = *(_QWORD *)(v21 + 8);
    if ( v31 )
    {
      v114 = v30;
      j_j___libc_free_0_0(v31);
      v30 = v114;
    }
  }
  *(_QWORD *)(v21 + 8) = v30;
  *(_DWORD *)(v21 + 16) = v28;
  if ( v135 > 0x40 && v134 )
    j_j___libc_free_0_0(v134);
  v112 = (unsigned __int64 *)(v21 + 24);
  if ( *(_DWORD *)(v21 + 32) <= 0x40u && (v32 = v131, v131 <= 0x40) )
  {
    v110 = v130;
    *(_DWORD *)(v21 + 32) = v131;
    *(_QWORD *)(v21 + 24) = v110 & (0xFFFFFFFFFFFFFFFFLL >> -v32);
  }
  else
  {
    sub_16A51C0((__int64)v112, (__int64)&v130);
    if ( v131 > 0x40 && v130 )
      j_j___libc_free_0_0(v130);
  }
LABEL_35:
  v111 = (__int64 *)(v21 + 8);
  v119 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF;
  if ( v119 == 1 )
    goto LABEL_4;
  v33 = 1;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v5 + 23) & 0x40) != 0 )
      v34 = *(_QWORD *)(v5 - 8);
    else
      v34 = v5 - 24LL * (*(_DWORD *)(v5 + 20) & 0xFFFFFFF);
    v35 = *(_QWORD *)(v34 + 24LL * v33);
    v36 = *(_BYTE *)(v35 + 16);
    if ( ((v36 - 5) & 0xFB) == 0 )
      goto LABEL_87;
    if ( v36 == 13 )
    {
      sub_16A5D10((__int64)&v128, v35 + 24, v123);
      v37 = v129;
      v38 = 1LL << ((unsigned __int8)v129 - 1);
      if ( v129 > 0x40 )
      {
        if ( (*(_QWORD *)(v128 + 8LL * ((v129 - 1) >> 6)) & v38) != 0 )
          v37 = sub_16A5810((__int64)&v128);
        else
          v37 = sub_16A57B0((__int64)&v128);
      }
      else if ( (v128 & v38) != 0 )
      {
        v37 = 64;
        if ( v128 << (64 - (unsigned __int8)v129) != -1 )
        {
          _BitScanReverse64(&v39, ~(v128 << (64 - (unsigned __int8)v129)));
          v37 = v39 ^ 0x3F;
        }
      }
      else if ( v128 )
      {
        _BitScanReverse64(&v53, v128);
        v37 = v129 - 64 + (v53 ^ 0x3F);
      }
      if ( (*(_DWORD *)v21 & 0x7FFFFFFFu) <= v37 )
        v37 = *(_DWORD *)v21 & 0x7FFFFFFF;
      *(_DWORD *)v21 = v37 & 0x7FFFFFFF | *(_DWORD *)v21 & 0x80000000;
      v40 = v129;
      v131 = v129;
      if ( v129 > 0x40 )
      {
        sub_16A4FD0((__int64)&v130, (const void **)&v128);
        v40 = v131;
        if ( v131 > 0x40 )
        {
          sub_16A8F40((__int64 *)&v130);
          v40 = v131;
          v42 = v130;
          goto LABEL_49;
        }
        v41 = v130;
      }
      else
      {
        v41 = v128;
      }
      v42 = ~v41 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v40);
      v130 = v42;
LABEL_49:
      v135 = v40;
      v134 = v42;
      v131 = 0;
      if ( *(_DWORD *)(v21 + 16) > 0x40u )
        sub_16A8890(v111, (__int64 *)&v134);
      else
        *(_QWORD *)(v21 + 8) &= v42;
      if ( v135 > 0x40 && v134 )
        j_j___libc_free_0_0(v134);
      if ( v131 > 0x40 && v130 )
        j_j___libc_free_0_0(v130);
      if ( *(_DWORD *)(v21 + 32) > 0x40u )
        sub_16A8890((__int64 *)v112, (__int64 *)&v128);
      else
        *(_QWORD *)(v21 + 24) &= v128;
      if ( v129 > 0x40 && v128 )
        j_j___libc_free_0_0(v128);
      goto LABEL_62;
    }
    v43 = *(_DWORD *)(a1 + 232);
    if ( !v43 )
    {
      ++*(_QWORD *)(a1 + 208);
      goto LABEL_156;
    }
    v44 = *(_QWORD *)(a1 + 216);
    v45 = (v43 - 1) & (((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4));
    v46 = (__int64 *)(v44 + 16LL * v45);
    v47 = *v46;
    if ( *v46 != v35 )
      break;
LABEL_68:
    v48 = *((_DWORD *)v46 + 2);
    if ( v48 >= 0 )
      goto LABEL_113;
    v49 = sub_1FDE940(a1, v48, v123);
    v50 = v49;
    if ( !v49 )
      goto LABEL_113;
    v51 = *(_DWORD *)v49 & 0x7FFFFFFF;
    if ( v51 > (*(_DWORD *)v21 & 0x7FFFFFFFu) )
      v51 = *(_DWORD *)v21 & 0x7FFFFFFF;
    v52 = *(_DWORD *)(v21 + 16) <= 0x40u;
    *(_DWORD *)v21 = v51 | *(_DWORD *)v21 & 0x80000000;
    if ( v52 )
      *(_QWORD *)(v21 + 8) &= *(_QWORD *)(v49 + 8);
    else
      sub_16A8890(v111, (__int64 *)(v49 + 8));
    if ( *(_DWORD *)(v21 + 32) > 0x40u )
      sub_16A8890((__int64 *)v112, (__int64 *)(v50 + 24));
    else
      *(_QWORD *)(v21 + 24) &= *(_QWORD *)(v50 + 24);
LABEL_62:
    if ( ++v33 == v119 )
      goto LABEL_4;
  }
  v68 = 1;
  v69 = 0;
  while ( v47 != -8 )
  {
    if ( !v69 && v47 == -16 )
      v69 = v46;
    v45 = (v43 - 1) & (v68 + v45);
    v46 = (__int64 *)(v44 + 16LL * v45);
    v47 = *v46;
    if ( v35 == *v46 )
      goto LABEL_68;
    ++v68;
  }
  v70 = v69;
  v71 = *(_DWORD *)(a1 + 224);
  if ( !v69 )
    v70 = v46;
  ++*(_QWORD *)(a1 + 208);
  v72 = v71 + 1;
  if ( 4 * (v71 + 1) < 3 * v43 )
  {
    if ( v43 - (v72 + *(_DWORD *)(a1 + 228)) > v43 >> 3 )
      goto LABEL_137;
    v121 = ((unsigned int)v35 >> 9) ^ ((unsigned int)v35 >> 4);
    v125 = v35;
    sub_1542080(v118, v43);
    v102 = *(_DWORD *)(a1 + 232);
    if ( v102 )
    {
      v103 = v102 - 1;
      v89 = 0;
      v104 = *(_QWORD *)(a1 + 216);
      v105 = 1;
      LODWORD(v106) = v103 & v121;
      v72 = *(_DWORD *)(a1 + 224) + 1;
      v35 = v125;
      v70 = (__int64 *)(v104 + 16LL * (v103 & v121));
      v107 = *v70;
      if ( *v70 != v125 )
      {
        while ( v107 != -8 )
        {
          if ( !v89 && v107 == -16 )
            v89 = v70;
          v106 = v103 & (unsigned int)(v106 + v105);
          v70 = (__int64 *)(v104 + 16 * v106);
          v107 = *v70;
          if ( v125 == *v70 )
            goto LABEL_137;
          ++v105;
        }
        goto LABEL_160;
      }
      goto LABEL_137;
    }
LABEL_224:
    ++*(_DWORD *)(a1 + 224);
    BUG();
  }
LABEL_156:
  v124 = v35;
  sub_1542080(v118, 2 * v43);
  v83 = *(_DWORD *)(a1 + 232);
  if ( !v83 )
    goto LABEL_224;
  v35 = v124;
  v84 = v83 - 1;
  v85 = *(_QWORD *)(a1 + 216);
  v72 = *(_DWORD *)(a1 + 224) + 1;
  LODWORD(v86) = v84 & (((unsigned int)v124 >> 9) ^ ((unsigned int)v124 >> 4));
  v70 = (__int64 *)(v85 + 16LL * (unsigned int)v86);
  v87 = *v70;
  if ( *v70 != v124 )
  {
    v88 = 1;
    v89 = 0;
    while ( v87 != -8 )
    {
      if ( v87 == -16 && !v89 )
        v89 = v70;
      v86 = v84 & (unsigned int)(v86 + v88);
      v70 = (__int64 *)(v85 + 16 * v86);
      v87 = *v70;
      if ( v124 == *v70 )
        goto LABEL_137;
      ++v88;
    }
LABEL_160:
    if ( v89 )
      v70 = v89;
  }
LABEL_137:
  *(_DWORD *)(a1 + 224) = v72;
  if ( *v70 != -8 )
    --*(_DWORD *)(a1 + 228);
  *v70 = v35;
  *((_DWORD *)v70 + 2) = 0;
LABEL_113:
  *(_BYTE *)(v21 + 3) &= ~0x80u;
LABEL_4:
  if ( v132[0] != (const __m128i *)v133 )
    _libc_free((unsigned __int64)v132[0]);
}
