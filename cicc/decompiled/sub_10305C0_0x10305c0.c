// Function: sub_10305C0
// Address: 0x10305c0
//
const __m128i **__fastcall sub_10305C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned int v5; // esi
  __int64 v6; // r9
  __int64 v7; // r8
  int v8; // r10d
  __int64 *v9; // r12
  __int64 v10; // rcx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  const __m128i **v13; // r12
  const __m128i *v14; // r15
  const __m128i *v15; // r14
  __int64 v16; // rax
  __int64 v17; // r8
  unsigned __int64 v18; // rdx
  const __m128i *v19; // r15
  __m128i *v20; // rdi
  unsigned __int64 v21; // rax
  const __m128i *i; // rsi
  __m128i v23; // xmm0
  unsigned __int64 v24; // rcx
  __m128i *v25; // rdx
  const __m128i *j; // rax
  __m128i v27; // xmm1
  unsigned int v28; // eax
  __int64 v29; // r8
  __int64 v30; // r9
  char v31; // r11
  __int64 v32; // rsi
  unsigned int v33; // eax
  __m128i **v34; // rdx
  __int64 v35; // rcx
  __m128i *v36; // r15
  __m128i **v37; // rax
  char v39; // dl
  __m128i *v40; // r13
  __m128i *v41; // rax
  __int64 v42; // rax
  char *v43; // rax
  __int64 v44; // rdx
  __int64 *m128i_i64; // rcx
  unsigned __int64 v46; // rax
  unsigned int v47; // ecx
  unsigned __int64 v48; // rax
  _QWORD *v49; // rax
  __int64 *v50; // rdx
  __int64 v51; // rcx
  __int64 v52; // rdi
  __int64 *v53; // rax
  char *v54; // rax
  __int64 v55; // rdx
  int v56; // eax
  int v57; // edx
  int v58; // r8d
  int v59; // r8d
  unsigned int v60; // eax
  __int64 v61; // rdi
  int v62; // esi
  __int64 *v63; // rcx
  int v64; // edi
  int v65; // edi
  __int64 v66; // r8
  int v67; // ecx
  unsigned int v68; // r14d
  __int64 v69; // rsi
  __int64 *v70; // rax
  unsigned __int64 v71; // [rsp+10h] [rbp-2A0h]
  unsigned int v72; // [rsp+18h] [rbp-298h]
  char v73; // [rsp+2Fh] [rbp-281h]
  __int64 v74; // [rsp+30h] [rbp-280h]
  __m128i *src; // [rsp+38h] [rbp-278h]
  void *srca; // [rsp+38h] [rbp-278h]
  void *srcb; // [rsp+38h] [rbp-278h]
  __m128i v78; // [rsp+40h] [rbp-270h] BYREF
  __m128i **v79; // [rsp+50h] [rbp-260h] BYREF
  __int64 v80; // [rsp+58h] [rbp-258h]
  _BYTE v81[256]; // [rsp+60h] [rbp-250h] BYREF
  __int64 v82; // [rsp+160h] [rbp-150h] BYREF
  __m128i **v83; // [rsp+168h] [rbp-148h]
  __int64 v84; // [rsp+170h] [rbp-140h]
  int v85; // [rsp+178h] [rbp-138h]
  char v86; // [rsp+17Ch] [rbp-134h]
  char v87; // [rsp+180h] [rbp-130h] BYREF

  v3 = a1 + 160;
  v5 = *(_DWORD *)(a1 + 184);
  if ( !v5 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_92;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 168);
  v8 = 1;
  v9 = 0;
  LODWORD(v10) = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v11 = (_QWORD *)(v7 + 40LL * (unsigned int)v10);
  v12 = *v11;
  if ( a2 != *v11 )
  {
    while ( v12 != -4096 )
    {
      if ( !v9 && v12 == -8192 )
        v9 = v11;
      v10 = (unsigned int)v6 & ((_DWORD)v10 + v8);
      v11 = (_QWORD *)(v7 + 40 * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        goto LABEL_3;
      ++v8;
    }
    if ( !v9 )
      v9 = v11;
    v56 = *(_DWORD *)(a1 + 176);
    ++*(_QWORD *)(a1 + 160);
    v57 = v56 + 1;
    if ( 4 * (v56 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 180) - v57 > v5 >> 3 )
      {
LABEL_87:
        *(_DWORD *)(a1 + 176) = v57;
        if ( *v9 != -4096 )
          --*(_DWORD *)(a1 + 180);
        *v9 = a2;
        v13 = (const __m128i **)(v9 + 1);
        *v13 = 0;
        v13[1] = 0;
        v13[2] = 0;
        *((_BYTE *)v13 + 24) = 0;
        goto LABEL_4;
      }
      sub_1030370(v3, v5);
      v64 = *(_DWORD *)(a1 + 184);
      if ( v64 )
      {
        v65 = v64 - 1;
        v66 = *(_QWORD *)(a1 + 168);
        v67 = 1;
        v68 = v65 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v9 = (__int64 *)(v66 + 40LL * v68);
        v69 = *v9;
        v57 = *(_DWORD *)(a1 + 176) + 1;
        v70 = 0;
        if ( a2 != *v9 )
        {
          while ( v69 != -4096 )
          {
            if ( !v70 && v69 == -8192 )
              v70 = v9;
            v6 = (unsigned int)(v67 + 1);
            v68 = v65 & (v67 + v68);
            v9 = (__int64 *)(v66 + 40LL * v68);
            v69 = *v9;
            if ( a2 == *v9 )
              goto LABEL_87;
            ++v67;
          }
          if ( v70 )
            v9 = v70;
        }
        goto LABEL_87;
      }
LABEL_117:
      ++*(_DWORD *)(a1 + 176);
      BUG();
    }
LABEL_92:
    sub_1030370(v3, 2 * v5);
    v58 = *(_DWORD *)(a1 + 184);
    if ( v58 )
    {
      v59 = v58 - 1;
      v6 = *(_QWORD *)(a1 + 168);
      v60 = v59 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = (__int64 *)(v6 + 40LL * v60);
      v57 = *(_DWORD *)(a1 + 176) + 1;
      v61 = *v9;
      if ( a2 != *v9 )
      {
        v62 = 1;
        v63 = 0;
        while ( v61 != -4096 )
        {
          if ( !v63 && v61 == -8192 )
            v63 = v9;
          v60 = v59 & (v62 + v60);
          v9 = (__int64 *)(v6 + 40LL * v60);
          v61 = *v9;
          if ( a2 == *v9 )
            goto LABEL_87;
          ++v62;
        }
        if ( v63 )
          v9 = v63;
      }
      goto LABEL_87;
    }
    goto LABEL_117;
  }
LABEL_3:
  v13 = (const __m128i **)(v11 + 1);
LABEL_4:
  v79 = (__m128i **)v81;
  v80 = 0x2000000000LL;
  v14 = v13[1];
  v15 = *v13;
  if ( *v13 == v14 )
  {
    v54 = (char *)sub_102DBD0(a1 + 288, *(_QWORD *)(a2 + 40));
    sub_1029A50((__int64)&v79, (char *)&v79[(unsigned int)v80], v54, &v54[8 * v55]);
  }
  else
  {
    if ( !*((_BYTE *)v13 + 24) )
      return v13;
    do
    {
      while ( (v15->m128i_i8[8] & 7) != 0 )
      {
        if ( v14 == ++v15 )
          goto LABEL_12;
      }
      v16 = (unsigned int)v80;
      v17 = v15->m128i_i64[0];
      v18 = (unsigned int)v80 + 1LL;
      if ( v18 > HIDWORD(v80) )
      {
        srcb = (void *)v15->m128i_i64[0];
        sub_C8D5F0((__int64)&v79, v81, v18, 8u, v17, v6);
        v16 = (unsigned int)v80;
        v17 = (__int64)srcb;
      }
      ++v15;
      v79[v16] = (__m128i *)v17;
      LODWORD(v80) = v80 + 1;
    }
    while ( v14 != v15 );
LABEL_12:
    v19 = v13[1];
    v20 = (__m128i *)*v13;
    if ( v19 != *v13 )
    {
      src = (__m128i *)*v13;
      _BitScanReverse64(&v21, v19 - v20);
      sub_1029840(v20, (__m128i *)v13[1], 2LL * (int)(63 - (v21 ^ 0x3F)));
      if ( (char *)v19 - (char *)v20 <= 256 )
      {
        sub_1029CF0(src, v19);
      }
      else
      {
        sub_1029CF0(src, src + 16);
        for ( i = src + 16; i != v19; *v25 = v23 )
        {
          v23 = _mm_loadu_si128(i);
          v24 = i->m128i_i64[0];
          v25 = (__m128i *)i;
          for ( j = i - 1; v24 < j->m128i_i64[0]; j[2] = v27 )
          {
            v27 = _mm_loadu_si128(j);
            v25 = (__m128i *)j--;
          }
          ++i;
        }
      }
    }
  }
  v28 = sub_CF5CA0(*(_QWORD *)(a1 + 256), a2);
  v86 = 1;
  v31 = 1;
  v32 = (__int64)&v78;
  v84 = 32;
  v85 = 0;
  v82 = 0;
  v73 = (((unsigned __int8)((v28 >> 6) | (v28 >> 4) | v28 | (v28 >> 2)) >> 1) ^ 1) & 1;
  v83 = (__m128i **)&v87;
  v33 = v80;
  v74 = ((char *)v13[1] - (char *)*v13) & 0xFFFFFFFF0LL;
  if ( !(_DWORD)v80 )
    goto LABEL_28;
  srca = (void *)a1;
  do
  {
    while ( 1 )
    {
      v34 = v79;
      v35 = v33;
      v36 = v79[v33 - 1];
      LODWORD(v80) = v33 - 1;
      if ( !v31 )
        goto LABEL_31;
      v37 = v83;
      v35 = HIDWORD(v84);
      v34 = &v83[HIDWORD(v84)];
      if ( v83 != v34 )
      {
        while ( v36 != *v37 )
        {
          if ( v34 == ++v37 )
            goto LABEL_45;
        }
        goto LABEL_25;
      }
LABEL_45:
      if ( HIDWORD(v84) < (unsigned int)v84 )
      {
        ++HIDWORD(v84);
        *v34 = v36;
        ++v82;
      }
      else
      {
LABEL_31:
        v32 = (__int64)v36;
        sub_C8CC70((__int64)&v82, (__int64)v36, (__int64)v34, v35, v29, v30);
        v31 = v86;
        if ( !v39 )
          goto LABEL_25;
      }
      v78 = (__m128i)(unsigned __int64)v36;
      v40 = (__m128i *)*v13;
      v32 = (__int64)(*v13)->m128i_i64 + v74;
      v41 = (__m128i *)sub_10297E0(*v13, v32, (unsigned __int64 *)&v78);
      if ( v41 != v40 )
      {
        if ( v36 == (__m128i *)v41[-1].m128i_i64[0] )
        {
          v40 = v41 - 1;
          if ( (__m128i *)v30 == &v41[-1] )
            break;
          goto LABEL_48;
        }
        v40 = v41;
      }
      if ( v40 == (__m128i *)v30 || v36 != (__m128i *)v40->m128i_i64[0] )
        break;
LABEL_48:
      v32 = v40->m128i_i64[1];
      if ( (v32 & 7) == 0 )
      {
        m128i_i64 = v36[3].m128i_i64;
        v29 = 0;
        v32 &= 0xFFFFFFFFFFFFFFF8LL;
        if ( v32 )
        {
          sub_1029DA0((__int64)srca + 224, v32, a2);
          m128i_i64 = (__int64 *)(v32 + 24);
          v29 = 0;
        }
        if ( (__int64 *)v36[3].m128i_i64[1] != m128i_i64 )
        {
          v32 = a2;
          v46 = sub_102AD20((__int64)srca, (unsigned __int8 *)a2, v73, m128i_i64, 0, (__int64)v36);
          v40->m128i_i64[1] = v46;
          v47 = v46 & 7;
          goto LABEL_53;
        }
        v42 = *(_QWORD *)(v36[4].m128i_i64[1] + 80);
        if ( !v42 )
        {
LABEL_41:
          v40->m128i_i64[1] = 0x2000000000000003LL;
LABEL_42:
          v43 = (char *)sub_102DBD0((__int64)srca + 288, (__int64)v36);
          v32 = (__int64)&v79[(unsigned int)v80];
          sub_1029A50((__int64)&v79, (char *)v32, v43, &v43[8 * v44]);
          goto LABEL_43;
        }
LABEL_39:
        if ( v36 == (__m128i *)(v42 - 24) )
        {
          if ( v40 )
          {
            v40->m128i_i64[1] = 0x4000000000000003LL;
            goto LABEL_43;
          }
          v46 = 0x4000000000000003LL;
          v47 = 3;
          goto LABEL_69;
        }
        if ( v40 )
          goto LABEL_41;
        goto LABEL_75;
      }
LABEL_25:
      v33 = v80;
      if ( !(_DWORD)v80 )
        goto LABEL_26;
    }
    v29 = 0;
    if ( &v36[3] != (__m128i *)v36[3].m128i_i64[1] )
    {
      v46 = sub_102AD20((__int64)srca, (unsigned __int8 *)a2, v73, (__m128i *)v36[3].m128i_i64, 0, (__int64)v36);
      v47 = v46 & 7;
      goto LABEL_69;
    }
    v40 = 0;
    v42 = *(_QWORD *)(v36[4].m128i_i64[1] + 80);
    if ( v42 )
      goto LABEL_39;
LABEL_75:
    v46 = 0x2000000000000003LL;
    v47 = 3;
LABEL_69:
    v78.m128i_i64[0] = (__int64)v36;
    v78.m128i_i64[1] = v46;
    v32 = (__int64)v13[1];
    if ( (const __m128i *)v32 == v13[2] )
    {
      v71 = v46;
      v72 = v47;
      sub_102D710(v13, (const __m128i *)v32, &v78);
      v46 = v71;
      v47 = v72;
    }
    else
    {
      if ( v32 )
      {
        *(__m128i *)v32 = _mm_loadu_si128(&v78);
        v32 = (__int64)v13[1];
      }
      v32 += 16;
      v13[1] = (const __m128i *)v32;
    }
LABEL_53:
    if ( v47 == 3 )
    {
      if ( v46 >> 61 == 1 )
        goto LABEL_42;
    }
    else
    {
      v48 = v46 & 0xFFFFFFFFFFFFFFF8LL;
      if ( v47 > 2 )
        BUG();
      v78.m128i_i64[0] = v48;
      if ( v48 )
      {
        v32 = (__int64)&v78;
        v49 = sub_1030100((__int64)srca + 224, v78.m128i_i64);
        v52 = (__int64)v49;
        if ( !*((_BYTE *)v49 + 28) )
          goto LABEL_63;
        v53 = (__int64 *)v49[1];
        v51 = *(unsigned int *)(v52 + 20);
        v50 = &v53[v51];
        if ( v53 == v50 )
        {
LABEL_60:
          if ( (unsigned int)v51 < *(_DWORD *)(v52 + 16) )
          {
            *(_DWORD *)(v52 + 20) = v51 + 1;
            *v50 = a2;
            ++*(_QWORD *)v52;
            goto LABEL_43;
          }
LABEL_63:
          v32 = a2;
          sub_C8CC70(v52, a2, (__int64)v50, v51, v29, v30);
          goto LABEL_43;
        }
        while ( a2 != *v53 )
        {
          if ( v50 == ++v53 )
            goto LABEL_60;
        }
      }
    }
LABEL_43:
    v33 = v80;
    v31 = v86;
  }
  while ( (_DWORD)v80 );
LABEL_26:
  if ( !v31 )
    _libc_free(v83, v32);
LABEL_28:
  if ( v79 != (__m128i **)v81 )
    _libc_free(v79, v32);
  return v13;
}
