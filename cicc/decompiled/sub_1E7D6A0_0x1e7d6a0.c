// Function: sub_1E7D6A0
// Address: 0x1e7d6a0
//
char __fastcall sub_1E7D6A0(__int64 a1, __int64 a2, _QWORD *a3, __int64 a4, char a5, int a6)
{
  _QWORD *v9; // r12
  bool v10; // zf
  unsigned __int64 v11; // rsi
  __m128i *v12; // rdi
  __m128i *v13; // rcx
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // edx
  __int64 v17; // rcx
  int v18; // edx
  unsigned int v19; // esi
  _QWORD *v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rdi
  unsigned int v23; // esi
  __int64 *v24; // rax
  __int64 v25; // r8
  __int64 *v26; // rbx
  __int64 v27; // r14
  unsigned int v28; // esi
  __int64 v29; // rdi
  __int64 v30; // rdx
  int v31; // r10d
  _QWORD *v32; // r9
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // rcx
  _QWORD *v35; // rcx
  _QWORD *v36; // r11
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __m128i *v40; // rsi
  char v41; // dl
  __int64 v42; // rax
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(__int64, __int64); // rdx
  __int16 v45; // dx
  __int64 v46; // rbx
  __int64 v47; // rdx
  int v48; // esi
  int v49; // eax
  int v50; // ecx
  int v51; // r10d
  __int64 v52; // rsi
  int v53; // r8d
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rdi
  int v56; // eax
  _QWORD *v57; // rdi
  unsigned int v58; // eax
  _QWORD *v59; // r9
  unsigned int v60; // eax
  __int64 v61; // rax
  __int64 v62; // r11
  unsigned int v63; // r9d
  int i; // eax
  int v65; // eax
  int v66; // edi
  int v67; // eax
  int v68; // ecx
  int v69; // r9d
  __int64 v70; // rsi
  int v71; // r8d
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // rdi
  unsigned int k; // eax
  _QWORD *v75; // rdi
  __int64 v76; // r10
  unsigned int v77; // eax
  int v78; // r8d
  int v79; // r10d
  __int64 *v80; // rax
  __int64 *j; // [rsp+10h] [rbp-50h]
  __int64 v84; // [rsp+10h] [rbp-50h]
  __int64 v85; // [rsp+18h] [rbp-48h]
  __m128i v86; // [rsp+20h] [rbp-40h] BYREF

  v9 = a3;
  v10 = *(_QWORD *)(a1 + 488) == 0;
  v86 = (__m128i)__PAIR128__(a4, (unsigned __int64)a3);
  if ( v10 )
  {
    v11 = *(unsigned int *)(a1 + 312);
    v12 = *(__m128i **)(a1 + 304);
    v13 = &v12[v11];
    if ( v12 != v13 )
    {
      v14 = (__int64)v12;
      while ( a3 != *(_QWORD **)v14 || a4 != *(_QWORD *)(v14 + 8) )
      {
        v14 += 16;
        if ( v13 == (__m128i *)v14 )
          goto LABEL_58;
      }
      if ( v13 != (__m128i *)v14 )
        goto LABEL_8;
    }
LABEL_58:
    if ( v11 > 7 )
    {
      while ( 1 )
      {
        sub_1E7AA70((_QWORD *)(a1 + 448), &v12[v11 - 1]);
        v49 = *(_DWORD *)(a1 + 312);
        *(_DWORD *)(a1 + 312) = v49 - 1;
        if ( v49 == 1 )
          break;
        v12 = *(__m128i **)(a1 + 304);
        v11 = (unsigned int)(v49 - 1);
      }
      sub_1E7AA70((_QWORD *)(a1 + 448), &v86);
    }
    else
    {
      if ( *(_DWORD *)(a1 + 312) >= *(_DWORD *)(a1 + 316) )
      {
        sub_16CD150(a1 + 304, (const void *)(a1 + 320), 0, 16, a5, a6);
        v13 = (__m128i *)(*(_QWORD *)(a1 + 304) + 16LL * *(unsigned int *)(a1 + 312));
      }
      *v13 = _mm_load_si128(&v86);
      ++*(_DWORD *)(a1 + 312);
    }
    goto LABEL_42;
  }
  LOBYTE(v14) = (unsigned __int8)sub_1E7AA70((_QWORD *)(a1 + 448), &v86);
  if ( v41 )
  {
LABEL_42:
    v42 = *(_QWORD *)(a2 + 16);
    if ( *(_WORD *)v42 == 15
      || ((v43 = *(_QWORD *)(a1 + 232),
           v44 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v43 + 136LL),
           v44 != sub_1DF74E0)
        ? (LOBYTE(v14) = v44(v43, a2))
        : (v45 = *(_WORD *)(a2 + 46), (v45 & 4) == 0) && (v45 & 8) != 0
        ? (LOBYTE(v14) = sub_1E15D00(a2, 0x4000000u, 2))
        : (v14 = (*(_QWORD *)(v42 + 8) >> 26) & 1LL),
          (_BYTE)v14) )
    {
      if ( !sub_1DD6970((__int64)v9, a4)
        || (sub_16AF710(&v86, dword_4FC8020, 0x64u),
            LODWORD(v14) = sub_1DF1780(*(_QWORD *)(a1 + 288), v9, a4),
            v86.m128i_i32[0] < (unsigned int)v14) )
      {
        v14 = *(unsigned int *)(a2 + 40);
        if ( !(_DWORD)v14 )
          return v14;
        v46 = 0;
        v47 = 40 * v14;
        while ( 1 )
        {
          v14 = v46 + *(_QWORD *)(a2 + 32);
          if ( !*(_BYTE *)v14 && (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
          {
            v48 = *(_DWORD *)(v14 + 8);
            if ( v48 < 0 )
            {
              v84 = v47;
              LOBYTE(v14) = sub_1E69E00(*(_QWORD *)(a1 + 248), v48);
              v47 = v84;
              if ( (_BYTE)v14 )
              {
                v14 = sub_1E69D00(*(_QWORD *)(a1 + 248), v48);
                v47 = v84;
                if ( *(_QWORD *)(v14 + 24) == *(_QWORD *)(a2 + 24) )
                  break;
              }
            }
          }
          v46 += 40;
          if ( v47 == v46 )
            return v14;
        }
      }
    }
  }
LABEL_8:
  if ( byte_4FC81E0 != 1 || v9 == (_QWORD *)a4 )
    return v14;
  v15 = *(_QWORD *)(a1 + 272);
  v16 = *(_DWORD *)(v15 + 256);
  if ( v16 )
  {
    v17 = *(_QWORD *)(v15 + 240);
    v18 = v16 - 1;
    v19 = v18 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v20 = (_QWORD *)(v17 + 16LL * v19);
    v21 = (_QWORD *)*v20;
    if ( v9 == (_QWORD *)*v20 )
    {
LABEL_12:
      v22 = v20[1];
    }
    else
    {
      v67 = 1;
      while ( v21 != (_QWORD *)-8LL )
      {
        v78 = v67 + 1;
        v19 = v18 & (v67 + v19);
        v20 = (_QWORD *)(v17 + 16LL * v19);
        v21 = (_QWORD *)*v20;
        if ( v9 == (_QWORD *)*v20 )
          goto LABEL_12;
        v67 = v78;
      }
      v22 = 0;
    }
    v23 = v18 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
    v24 = (__int64 *)(v17 + 16LL * v23);
    v25 = *v24;
    if ( a4 == *v24 )
    {
      if ( v24[1] == v22 )
      {
LABEL_79:
        v61 = v24[1];
        if ( v61 )
        {
          v14 = *(_QWORD *)(v61 + 32);
          if ( a4 == *(_QWORD *)v14 )
            return v14;
        }
      }
    }
    else
    {
      v62 = *v24;
      v63 = v18 & (((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4));
      for ( i = 1; ; i = v79 )
      {
        if ( v62 == -8 )
        {
          if ( v22 )
            goto LABEL_15;
          goto LABEL_89;
        }
        v79 = i + 1;
        v63 = v18 & (i + v63);
        v80 = (__int64 *)(v17 + 16LL * v63);
        v62 = *v80;
        if ( a4 == *v80 )
          break;
      }
      if ( v80[1] != v22 )
        goto LABEL_15;
LABEL_89:
      v65 = 1;
      while ( v25 != -8 )
      {
        v66 = v65 + 1;
        v23 = v18 & (v65 + v23);
        v24 = (__int64 *)(v17 + 16LL * v23);
        v25 = *v24;
        if ( a4 == *v24 )
          goto LABEL_79;
        v65 = v66;
      }
    }
  }
LABEL_15:
  if ( !a5 )
  {
    v26 = *(__int64 **)(a4 + 64);
    for ( j = *(__int64 **)(a4 + 72); j != v26; ++v26 )
    {
      v27 = *v26;
      if ( v9 != (_QWORD *)*v26 )
      {
        v85 = *(_QWORD *)(a1 + 256);
        sub_1E06620(v85);
        LOBYTE(v14) = sub_1E05550(*(_QWORD *)(v85 + 1312), a4, v27);
        if ( !(_BYTE)v14 )
          return v14;
      }
    }
  }
  v28 = *(_DWORD *)(a1 + 520);
  v86.m128i_i64[0] = (__int64)v9;
  v29 = a1 + 496;
  v86.m128i_i64[1] = a4;
  if ( !v28 )
  {
    ++*(_QWORD *)(a1 + 496);
    goto LABEL_70;
  }
  v30 = *(_QWORD *)(a1 + 504);
  v31 = 1;
  v32 = 0;
  v33 = (((((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
         | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)) << 32)) >> 22)
      ^ ((((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)
        | ((unsigned __int64)(((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned int)a4 >> 9) ^ ((unsigned int)a4 >> 4)) << 32));
  v34 = ((9 * (((v33 - 1 - (v33 << 13)) >> 8) ^ (v33 - 1 - (v33 << 13)))) >> 15)
      ^ (9 * (((v33 - 1 - (v33 << 13)) >> 8) ^ (v33 - 1 - (v33 << 13))));
  for ( LODWORD(v14) = (v28 - 1) & (((v34 - 1 - (v34 << 27)) >> 31) ^ (v34 - 1 - ((_DWORD)v34 << 27)));
        ;
        LODWORD(v14) = (v28 - 1) & v37 )
  {
    v35 = (_QWORD *)(v30 + 16LL * (unsigned int)v14);
    v36 = (_QWORD *)*v35;
    if ( v9 == (_QWORD *)*v35 && a4 == v35[1] )
      return v14;
    if ( v36 == (_QWORD *)-8LL )
      break;
    if ( v36 == (_QWORD *)-16LL && v35[1] == -16 && !v32 )
      v32 = (_QWORD *)(v30 + 16LL * (unsigned int)v14);
LABEL_28:
    v37 = v31 + v14;
    ++v31;
  }
  if ( v35[1] != -8 )
    goto LABEL_28;
  v38 = *(_DWORD *)(a1 + 512);
  if ( v32 )
    v35 = v32;
  ++*(_QWORD *)(a1 + 496);
  v39 = v38 + 1;
  if ( 4 * v39 < 3 * v28 )
  {
    if ( v28 - *(_DWORD *)(a1 + 516) - v39 > v28 >> 3 )
      goto LABEL_34;
    sub_1E7D400(v29, v28);
    v68 = *(_DWORD *)(a1 + 520);
    if ( v68 )
    {
      v9 = (_QWORD *)v86.m128i_i64[0];
      v69 = v68 - 1;
      v35 = 0;
      v71 = 1;
      v72 = (((((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)
             | ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[0] >> 9) ^ ((unsigned __int32)v86.m128i_i32[0] >> 4)) << 32))
            - 1
            - ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)) << 32)) >> 22)
          ^ ((((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)
            | ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[0] >> 9) ^ ((unsigned __int32)v86.m128i_i32[0] >> 4)) << 32))
           - 1
           - ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)) << 32));
      v73 = ((9 * (((v72 - 1 - (v72 << 13)) >> 8) ^ (v72 - 1 - (v72 << 13)))) >> 15)
          ^ (9 * (((v72 - 1 - (v72 << 13)) >> 8) ^ (v72 - 1 - (v72 << 13))));
      for ( k = v69 & (((v73 - 1 - (v73 << 27)) >> 31) ^ (v73 - 1 - ((_DWORD)v73 << 27))); ; k = v69 & v77 )
      {
        v70 = *(_QWORD *)(a1 + 504);
        v75 = (_QWORD *)(v70 + 16LL * k);
        v76 = *v75;
        if ( *(_OWORD *)v75 == *(_OWORD *)&v86 )
        {
          v35 = (_QWORD *)(v70 + 16LL * k);
          v39 = *(_DWORD *)(a1 + 512) + 1;
          goto LABEL_34;
        }
        if ( v76 == -8 )
        {
          if ( v75[1] == -8 )
          {
            v39 = *(_DWORD *)(a1 + 512) + 1;
            if ( !v35 )
              v35 = v75;
            goto LABEL_34;
          }
        }
        else if ( v76 == -16 && v75[1] == -16 && !v35 )
        {
          v35 = (_QWORD *)(v70 + 16LL * k);
        }
        v77 = v71 + k;
        ++v71;
      }
    }
LABEL_122:
    ++*(_DWORD *)(a1 + 512);
    BUG();
  }
LABEL_70:
  sub_1E7D400(v29, 2 * v28);
  v50 = *(_DWORD *)(a1 + 520);
  if ( !v50 )
    goto LABEL_122;
  v9 = (_QWORD *)v86.m128i_i64[0];
  v51 = v50 - 1;
  v53 = 1;
  v54 = (((((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)
         | ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[0] >> 9) ^ ((unsigned __int32)v86.m128i_i32[0] >> 4)) << 32))
        - 1
        - ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)) << 32)) >> 22)
      ^ ((((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)
        | ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[0] >> 9) ^ ((unsigned __int32)v86.m128i_i32[0] >> 4)) << 32))
       - 1
       - ((unsigned __int64)(((unsigned __int32)v86.m128i_i32[2] >> 9) ^ ((unsigned __int32)v86.m128i_i32[2] >> 4)) << 32));
  v55 = ((9 * (((v54 - 1 - (v54 << 13)) >> 8) ^ (v54 - 1 - (v54 << 13)))) >> 15)
      ^ (9 * (((v54 - 1 - (v54 << 13)) >> 8) ^ (v54 - 1 - (v54 << 13))));
  v56 = ((v55 - 1 - (v55 << 27)) >> 31) ^ (v55 - 1 - ((_DWORD)v55 << 27));
  v57 = 0;
  v58 = (v50 - 1) & v56;
  while ( 2 )
  {
    v52 = *(_QWORD *)(a1 + 504);
    v35 = (_QWORD *)(v52 + 16LL * v58);
    v59 = (_QWORD *)*v35;
    if ( *(_OWORD *)v35 == *(_OWORD *)&v86 )
    {
      v39 = *(_DWORD *)(a1 + 512) + 1;
      goto LABEL_34;
    }
    if ( v59 != (_QWORD *)-8LL )
    {
      if ( v59 == (_QWORD *)-16LL && v35[1] == -16 && !v57 )
        v57 = (_QWORD *)(v52 + 16LL * v58);
      goto LABEL_78;
    }
    if ( v35[1] != -8 )
    {
LABEL_78:
      v60 = v53 + v58;
      ++v53;
      v58 = v51 & v60;
      continue;
    }
    break;
  }
  v39 = *(_DWORD *)(a1 + 512) + 1;
  if ( v57 )
    v35 = v57;
LABEL_34:
  *(_DWORD *)(a1 + 512) = v39;
  if ( *v35 != -8 || v35[1] != -8 )
    --*(_DWORD *)(a1 + 516);
  *v35 = v9;
  LOBYTE(v14) = v86.m128i_i8[8];
  v35[1] = v86.m128i_i64[1];
  v40 = *(__m128i **)(a1 + 536);
  if ( v40 == *(__m128i **)(a1 + 544) )
  {
    LOBYTE(v14) = sub_1E7D280((const __m128i **)(a1 + 528), v40, &v86);
  }
  else
  {
    if ( v40 )
    {
      *v40 = _mm_load_si128(&v86);
      v40 = *(__m128i **)(a1 + 536);
    }
    *(_QWORD *)(a1 + 536) = v40 + 1;
  }
  return v14;
}
