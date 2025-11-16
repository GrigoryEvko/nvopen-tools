// Function: sub_16374B0
// Address: 0x16374b0
//
__int64 __fastcall sub_16374B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rdi
  unsigned int v7; // esi
  __int64 v8; // rdx
  int v9; // r14d
  unsigned int v10; // r8d
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int64 *v14; // rcx
  unsigned int i; // r10d
  __int64 *v16; // r8
  __int64 v17; // r9
  unsigned int v18; // r10d
  __int64 v19; // rax
  int v21; // edx
  int v22; // r8d
  __int64 v23; // rdx
  __int64 v24; // rsi
  unsigned int v25; // ecx
  __int64 *v26; // rax
  __int64 v27; // r8
  __int64 v28; // r15
  unsigned int v29; // esi
  __int64 v30; // r9
  __int64 v31; // rdi
  unsigned int v32; // ecx
  _QWORD *v33; // r14
  __int64 v34; // rdx
  _QWORD *v35; // r8
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  int v41; // r9d
  unsigned int v42; // edi
  unsigned __int64 v43; // rsi
  unsigned __int64 v44; // rsi
  unsigned int m; // eax
  _QWORD *v46; // rsi
  unsigned int v47; // eax
  __int64 v48; // rax
  __m128i *v49; // rdx
  __int64 v50; // r14
  __m128i si128; // xmm0
  __int64 v52; // rax
  size_t v53; // rdx
  _DWORD *v54; // rdi
  const char *v55; // rsi
  unsigned __int64 v56; // rax
  _BYTE *v57; // rdi
  _BYTE *v58; // rax
  size_t v59; // rdx
  const char *v60; // rsi
  __int64 v61; // rax
  __int64 v62; // rax
  int v63; // ecx
  __int64 v64; // rdx
  int v65; // r8d
  __int64 *v66; // r9
  unsigned __int64 v67; // rsi
  unsigned __int64 v68; // rsi
  unsigned int j; // eax
  __int64 v70; // rsi
  unsigned int v71; // eax
  int v72; // edx
  int v73; // edx
  int v74; // r8d
  unsigned int k; // eax
  __int64 v76; // rsi
  unsigned int v77; // eax
  __int64 v78; // rax
  int v79; // r11d
  _QWORD *v80; // r10
  int v81; // edi
  int v82; // ecx
  int v83; // eax
  int v84; // r9d
  int v85; // eax
  int v86; // edx
  __int64 v87; // rsi
  unsigned int v88; // eax
  __int64 v89; // rdi
  _QWORD *v90; // r8
  int v91; // edx
  int v92; // edx
  __int64 v93; // rdi
  unsigned int v94; // eax
  __int64 v95; // rsi
  int v96; // r10d
  __int64 *v97; // r11
  _QWORD *v98; // [rsp+8h] [rbp-48h]
  int v99; // [rsp+8h] [rbp-48h]
  __int64 v100; // [rsp+8h] [rbp-48h]
  size_t v101; // [rsp+8h] [rbp-48h]
  unsigned int v102; // [rsp+8h] [rbp-48h]
  __int64 v103[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 64;
  v7 = *(_DWORD *)(a1 + 88);
  if ( !v7 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_59;
  }
  v8 = *(_QWORD *)(a1 + 72);
  v9 = 1;
  v10 = (unsigned int)a3 >> 9;
  v11 = (((v10 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(v10 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
      ^ ((v10 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(v10 ^ ((unsigned int)a3 >> 4)) << 32));
  v12 = ((9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13)))) >> 15)
      ^ (9 * (((v11 - 1 - (v11 << 13)) >> 8) ^ (v11 - 1 - (v11 << 13))));
  v13 = ((v12 - 1 - (v12 << 27)) >> 31) ^ (v12 - 1 - (v12 << 27));
  v14 = 0;
  for ( i = v13 & (v7 - 1); ; i = (v7 - 1) & v18 )
  {
    v16 = (__int64 *)(v8 + 24LL * i);
    v17 = *v16;
    if ( *v16 == a2 && a3 == v16[1] )
    {
      v19 = v16[2];
      return *(_QWORD *)(v19 + 24);
    }
    if ( v17 == -8 )
      break;
    if ( v17 == -16 && v16[1] == -16 && !v14 )
      v14 = (__int64 *)(v8 + 24LL * i);
LABEL_9:
    v18 = v9 + i;
    ++v9;
  }
  if ( v16[1] != -8 )
    goto LABEL_9;
  v21 = *(_DWORD *)(a1 + 80);
  if ( !v14 )
    v14 = v16;
  ++*(_QWORD *)(a1 + 64);
  v22 = v21 + 1;
  if ( 4 * (v21 + 1) >= 3 * v7 )
  {
LABEL_59:
    sub_1637200(v6, 2 * v7);
    v63 = *(_DWORD *)(a1 + 88);
    if ( v63 )
    {
      v65 = 1;
      v66 = 0;
      v6 = (unsigned int)(v63 - 1);
      v67 = (((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
             | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
          ^ ((((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
            | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32));
      v68 = ((9 * (((v67 - 1 - (v67 << 13)) >> 8) ^ (v67 - 1 - (v67 << 13)))) >> 15)
          ^ (9 * (((v67 - 1 - (v67 << 13)) >> 8) ^ (v67 - 1 - (v67 << 13))));
      for ( j = v6 & (((v68 - 1 - (v68 << 27)) >> 31) ^ (v68 - 1 - ((_DWORD)v68 << 27))); ; j = v6 & v71 )
      {
        v64 = *(_QWORD *)(a1 + 72);
        v14 = (__int64 *)(v64 + 24LL * j);
        v70 = *v14;
        if ( *v14 == a2 && a3 == v14[1] )
          break;
        if ( v70 == -8 )
        {
          if ( v14[1] == -8 )
          {
LABEL_115:
            if ( v66 )
              v14 = v66;
            v22 = *(_DWORD *)(a1 + 80) + 1;
            goto LABEL_18;
          }
        }
        else if ( v70 == -16 && v14[1] == -16 && !v66 )
        {
          v66 = (__int64 *)(v64 + 24LL * j);
        }
        v71 = v65 + j;
        ++v65;
      }
      goto LABEL_107;
    }
LABEL_127:
    ++*(_DWORD *)(a1 + 80);
    BUG();
  }
  if ( v7 - *(_DWORD *)(a1 + 84) - v22 <= v7 >> 3 )
  {
    v99 = v13;
    sub_1637200(v6, v7);
    v72 = *(_DWORD *)(a1 + 88);
    if ( v72 )
    {
      v73 = v72 - 1;
      v6 = *(_QWORD *)(a1 + 72);
      v66 = 0;
      v74 = 1;
      for ( k = v73 & v99; ; k = v73 & v77 )
      {
        v14 = (__int64 *)(v6 + 24LL * k);
        v76 = *v14;
        if ( *v14 == a2 && a3 == v14[1] )
          break;
        if ( v76 == -8 )
        {
          if ( v14[1] == -8 )
            goto LABEL_115;
        }
        else if ( v76 == -16 && v14[1] == -16 && !v66 )
        {
          v66 = (__int64 *)(v6 + 24LL * k);
        }
        v77 = v74 + k;
        ++v74;
      }
LABEL_107:
      v22 = *(_DWORD *)(a1 + 80) + 1;
      goto LABEL_18;
    }
    goto LABEL_127;
  }
LABEL_18:
  *(_DWORD *)(a1 + 80) = v22;
  if ( *v14 != -8 || v14[1] != -8 )
    --*(_DWORD *)(a1 + 84);
  *v14 = a2;
  v14[1] = a3;
  v14[2] = 0;
  v23 = *(unsigned int *)(a1 + 24);
  v24 = *(_QWORD *)(a1 + 8);
  if ( !(_DWORD)v23 )
    goto LABEL_33;
  v6 = (unsigned int)(v23 - 1);
  v25 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( *v26 != a2 )
  {
    v83 = 1;
    while ( v27 != -8 )
    {
      v84 = v83 + 1;
      v25 = v6 & (v83 + v25);
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( *v26 == a2 )
        goto LABEL_22;
      v83 = v84;
    }
LABEL_33:
    v23 *= 16;
    v28 = *(_QWORD *)(v24 + v23 + 8);
    if ( *(_BYTE *)(a1 + 96) )
      goto LABEL_34;
    goto LABEL_23;
  }
LABEL_22:
  v28 = v26[1];
  if ( !*(_BYTE *)(a1 + 96) )
    goto LABEL_23;
LABEL_34:
  v48 = sub_16BA580(v6, v24, v23);
  v49 = *(__m128i **)(v48 + 24);
  v50 = v48;
  if ( *(_QWORD *)(v48 + 16) - (_QWORD)v49 <= 0x11u )
  {
    v50 = sub_16E7EE0(v48, "Running analysis: ", 18);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_42ABC60);
    v49[1].m128i_i16[0] = 8250;
    *v49 = si128;
    *(_QWORD *)(v48 + 24) += 18LL;
  }
  v52 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v28 + 24LL))(v28);
  v54 = *(_DWORD **)(v50 + 24);
  v55 = (const char *)v52;
  v56 = *(_QWORD *)(v50 + 16) - (_QWORD)v54;
  if ( v53 > v56 )
  {
    v62 = sub_16E7EE0(v50, v55);
    v54 = *(_DWORD **)(v62 + 24);
    v50 = v62;
    v56 = *(_QWORD *)(v62 + 16) - (_QWORD)v54;
  }
  else if ( v53 )
  {
    v101 = v53;
    memcpy(v54, v55, v53);
    v78 = *(_QWORD *)(v50 + 16);
    v54 = (_DWORD *)(v101 + *(_QWORD *)(v50 + 24));
    *(_QWORD *)(v50 + 24) = v54;
    v56 = v78 - (_QWORD)v54;
  }
  if ( v56 <= 3 )
  {
    v61 = sub_16E7EE0(v50, " on ", 4);
    v57 = *(_BYTE **)(v61 + 24);
    v50 = v61;
  }
  else
  {
    *v54 = 544108320;
    v57 = (_BYTE *)(*(_QWORD *)(v50 + 24) + 4LL);
    *(_QWORD *)(v50 + 24) = v57;
  }
  v58 = *(_BYTE **)(v50 + 16);
  v59 = *(_QWORD *)(a3 + 184);
  v60 = *(const char **)(a3 + 176);
  if ( v59 > v58 - v57 )
  {
    v50 = sub_16E7EE0(v50, v60);
    v58 = *(_BYTE **)(v50 + 16);
    v57 = *(_BYTE **)(v50 + 24);
  }
  else if ( v59 )
  {
    v100 = *(_QWORD *)(a3 + 184);
    memcpy(v57, v60, v59);
    v58 = *(_BYTE **)(v50 + 16);
    v57 = (_BYTE *)(v100 + *(_QWORD *)(v50 + 24));
    *(_QWORD *)(v50 + 24) = v57;
  }
  if ( v57 == v58 )
  {
    sub_16E7EE0(v50, "\n", 1);
  }
  else
  {
    *v57 = 10;
    ++*(_QWORD *)(v50 + 24);
  }
LABEL_23:
  v29 = *(_DWORD *)(a1 + 56);
  v30 = a1 + 32;
  if ( !v29 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_91;
  }
  v31 = *(_QWORD *)(a1 + 40);
  v32 = (v29 - 1) & (((unsigned int)a3 >> 4) ^ ((unsigned int)a3 >> 9));
  v33 = (_QWORD *)(v31 + 32LL * v32);
  v34 = *v33;
  if ( a3 == *v33 )
  {
    v35 = v33 + 1;
    goto LABEL_26;
  }
  v79 = 1;
  v80 = 0;
  while ( v34 != -8 )
  {
    if ( v34 != -16 || v80 )
      v33 = v80;
    v96 = v79 + 1;
    v32 = (v29 - 1) & (v79 + v32);
    v97 = (__int64 *)(v31 + 32LL * v32);
    v34 = *v97;
    if ( a3 == *v97 )
    {
      v35 = v97 + 1;
      v33 = (_QWORD *)(v31 + 32LL * v32);
      goto LABEL_26;
    }
    v79 = v96;
    v80 = v33;
    v33 = (_QWORD *)(v31 + 32LL * v32);
  }
  v81 = *(_DWORD *)(a1 + 48);
  if ( v80 )
    v33 = v80;
  ++*(_QWORD *)(a1 + 32);
  v82 = v81 + 1;
  if ( 4 * (v81 + 1) >= 3 * v29 )
  {
LABEL_91:
    sub_1636F80(a1 + 32, 2 * v29);
    v85 = *(_DWORD *)(a1 + 56);
    if ( v85 )
    {
      v86 = v85 - 1;
      v87 = *(_QWORD *)(a1 + 40);
      v88 = (v85 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v82 = *(_DWORD *)(a1 + 48) + 1;
      v33 = (_QWORD *)(v87 + 32LL * v88);
      v89 = *v33;
      if ( a3 == *v33 )
        goto LABEL_83;
      v30 = 1;
      v90 = 0;
      while ( v89 != -8 )
      {
        if ( !v90 && v89 == -16 )
          v90 = v33;
        v88 = v86 & (v30 + v88);
        v33 = (_QWORD *)(v87 + 32LL * v88);
        v89 = *v33;
        if ( a3 == *v33 )
          goto LABEL_83;
        v30 = (unsigned int)(v30 + 1);
      }
LABEL_95:
      if ( v90 )
        v33 = v90;
      goto LABEL_83;
    }
LABEL_128:
    ++*(_DWORD *)(a1 + 48);
    BUG();
  }
  if ( v29 - *(_DWORD *)(a1 + 52) - v82 <= v29 >> 3 )
  {
    v102 = ((unsigned int)a3 >> 4) ^ ((unsigned int)a3 >> 9);
    sub_1636F80(a1 + 32, v29);
    v91 = *(_DWORD *)(a1 + 56);
    if ( v91 )
    {
      v92 = v91 - 1;
      v93 = *(_QWORD *)(a1 + 40);
      v30 = 1;
      v90 = 0;
      v94 = v92 & v102;
      v82 = *(_DWORD *)(a1 + 48) + 1;
      v33 = (_QWORD *)(v93 + 32LL * (v92 & v102));
      v95 = *v33;
      if ( a3 == *v33 )
        goto LABEL_83;
      while ( v95 != -8 )
      {
        if ( v95 == -16 && !v90 )
          v90 = v33;
        v94 = v92 & (v30 + v94);
        v33 = (_QWORD *)(v93 + 32LL * v94);
        v95 = *v33;
        if ( a3 == *v33 )
          goto LABEL_83;
        v30 = (unsigned int)(v30 + 1);
      }
      goto LABEL_95;
    }
    goto LABEL_128;
  }
LABEL_83:
  *(_DWORD *)(a1 + 48) = v82;
  if ( *v33 != -8 )
    --*(_DWORD *)(a1 + 52);
  v35 = v33 + 1;
  *v33 = a3;
  v33[2] = v33 + 1;
  v33[1] = v33 + 1;
  v33[3] = 0;
LABEL_26:
  v98 = v35;
  (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64, _QWORD *, __int64))(*(_QWORD *)v28 + 16LL))(
    v103,
    v28,
    a3,
    a1,
    v35,
    v30);
  v36 = sub_22077B0(32);
  *(_QWORD *)(v36 + 16) = a2;
  v37 = v103[0];
  v103[0] = 0;
  *(_QWORD *)(v36 + 24) = v37;
  sub_2208C80(v36, v98);
  v38 = v103[0];
  ++v33[3];
  if ( v38 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v38 + 8LL))(v38);
  v39 = *(unsigned int *)(a1 + 88);
  v40 = *(_QWORD *)(a1 + 72);
  if ( (_DWORD)v39 )
  {
    v41 = 1;
    v42 = (unsigned int)a3 >> 9;
    v43 = (((v42 ^ ((unsigned int)a3 >> 4)
           | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
          - 1
          - ((unsigned __int64)(v42 ^ ((unsigned int)a3 >> 4)) << 32)) >> 22)
        ^ ((v42 ^ ((unsigned int)a3 >> 4) | ((unsigned __int64)(((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)) << 32))
         - 1
         - ((unsigned __int64)(v42 ^ ((unsigned int)a3 >> 4)) << 32));
    v44 = ((9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13)))) >> 15)
        ^ (9 * (((v43 - 1 - (v43 << 13)) >> 8) ^ (v43 - 1 - (v43 << 13))));
    for ( m = (v39 - 1) & (((v44 - 1 - (v44 << 27)) >> 31) ^ (v44 - 1 - ((_DWORD)v44 << 27))); ; m = (v39 - 1) & v47 )
    {
      v46 = (_QWORD *)(v40 + 24LL * m);
      if ( *v46 == a2 && a3 == v46[1] )
        break;
      if ( *v46 == -8 && v46[1] == -8 )
        goto LABEL_47;
      v47 = v41 + m;
      ++v41;
    }
  }
  else
  {
LABEL_47:
    v46 = (_QWORD *)(v40 + 24 * v39);
  }
  v19 = v33[2];
  v46[2] = v19;
  return *(_QWORD *)(v19 + 24);
}
