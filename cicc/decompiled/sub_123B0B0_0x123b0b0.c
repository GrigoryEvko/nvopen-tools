// Function: sub_123B0B0
// Address: 0x123b0b0
//
__int64 __fastcall sub_123B0B0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r12d
  __int64 i; // r12
  __int64 v7; // r15
  __int64 v8; // r8
  bool v9; // zf
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // r14
  const __m128i *v19; // r11
  const __m128i *v20; // rax
  unsigned __int64 v21; // r15
  __int64 v22; // rax
  __int64 v23; // rdi
  __m128i *v24; // rdx
  __m128i *v25; // r11
  __int64 v26; // rdx
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rax
  const __m128i *v31; // r15
  __int64 *v32; // rcx
  const __m128i *v33; // r13
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rax
  __int64 v37; // r13
  __int64 v38; // r12
  const char *v39; // rax
  unsigned __int64 v40; // rsi
  int *v41; // rax
  int *v42; // rsi
  __int32 v43; // eax
  __m128i *v44; // r9
  __int64 v45; // rax
  __int64 v46; // [rsp+28h] [rbp-D8h]
  __int64 v47; // [rsp+28h] [rbp-D8h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  __int64 *v49; // [rsp+38h] [rbp-C8h]
  char v50; // [rsp+4Fh] [rbp-B1h] BYREF
  unsigned int v51; // [rsp+50h] [rbp-B0h] BYREF
  int v52; // [rsp+54h] [rbp-ACh] BYREF
  __int64 v53; // [rsp+58h] [rbp-A8h] BYREF
  __m128i v54; // [rsp+60h] [rbp-A0h] BYREF
  __m128i v55; // [rsp+70h] [rbp-90h] BYREF
  __int64 v56; // [rsp+80h] [rbp-80h]
  unsigned __int64 v57; // [rsp+88h] [rbp-78h]
  char v58; // [rsp+90h] [rbp-70h]
  char v59; // [rsp+91h] [rbp-6Fh]
  __int64 v60; // [rsp+A0h] [rbp-60h] BYREF
  int v61; // [rsp+A8h] [rbp-58h] BYREF
  int *v62; // [rsp+B0h] [rbp-50h]
  int *v63; // [rsp+B8h] [rbp-48h]
  int *v64; // [rsp+C0h] [rbp-40h]
  __int64 v65; // [rsp+C8h] [rbp-38h]

  *(_DWORD *)(a1 + 240) = sub_1205200(a1 + 176);
  if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':' in calls")
    || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in calls") )
  {
    return 1;
  }
  v61 = 0;
  v63 = &v61;
  v64 = &v61;
  v62 = 0;
  v65 = 0;
  for ( i = a1 + 176; ; *(_DWORD *)(a1 + 240) = sub_1205200(i) )
  {
    v54.m128i_i64[0] = 0;
    if ( (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in call") )
      goto LABEL_56;
    if ( (unsigned __int8)sub_120AFE0(a1, 440, "expected 'callee' in call") )
      goto LABEL_56;
    if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") )
      goto LABEL_56;
    v7 = *(_QWORD *)(a1 + 232);
    if ( (unsigned __int8)sub_12122D0(a1, &v54, &v51) )
      goto LABEL_56;
    v9 = *(_DWORD *)(a1 + 240) == 4;
    v50 = 0;
    v52 = 0;
    LODWORD(v53) = 0;
    if ( v9 )
    {
      do
      {
        v35 = sub_1205200(i);
        *(_DWORD *)(a1 + 240) = v35;
        switch ( v35 )
        {
          case 443:
            *(_DWORD *)(a1 + 240) = sub_1205200(i);
            if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_12114C0(a1, &v50) )
              goto LABEL_56;
            break;
          case 446:
            *(_DWORD *)(a1 + 240) = sub_1205200(i);
            if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_120BD00(a1, &v52) )
              goto LABEL_56;
            break;
          case 60:
            *(_DWORD *)(a1 + 240) = sub_1205200(i);
            if ( (unsigned __int8)sub_120AFE0(a1, 16, "expected ':'") || (unsigned __int8)sub_1210F40(a1, &v53) )
              goto LABEL_56;
            break;
          default:
            v59 = 1;
            v36 = a1;
            v37 = i;
            v38 = v36;
            v39 = "expected hotness, relbf, or tail";
            goto LABEL_55;
        }
      }
      while ( *(_DWORD *)(a1 + 240) == 4 );
      if ( v50 && v52 )
      {
        v45 = a1;
        v59 = 1;
        v37 = i;
        v38 = v45;
        v39 = "Expected only one of hotness or relbf";
LABEL_55:
        v40 = *(_QWORD *)(v38 + 232);
        v58 = 3;
        v55.m128i_i64[0] = (__int64)v39;
        sub_11FD800(v37, v40, (__int64)&v55, 1);
LABEL_56:
        v3 = 1;
        goto LABEL_47;
      }
    }
    if ( (v54.m128i_i64[0] & 0xFFFFFFFFFFFFFFF8LL) == 0xFFFFFFFFFFFFFFF8LL )
    {
      v41 = v62;
      if ( v62 )
      {
        v42 = &v61;
        do
        {
          if ( v41[8] < v51 )
          {
            v41 = (int *)*((_QWORD *)v41 + 3);
          }
          else
          {
            v42 = v41;
            v41 = (int *)*((_QWORD *)v41 + 2);
          }
        }
        while ( v41 );
        if ( v42 != &v61 && v51 >= v42[8] )
        {
LABEL_66:
          v43 = *(_DWORD *)(a2 + 8);
          v55.m128i_i64[1] = v7;
          v55.m128i_i32[0] = v43;
          v44 = (__m128i *)*((_QWORD *)v42 + 6);
          if ( v44 == *((__m128i **)v42 + 7) )
          {
            sub_12171B0((const __m128i **)v42 + 5, *((const __m128i **)v42 + 6), &v55);
          }
          else
          {
            if ( v44 )
            {
              *v44 = _mm_load_si128(&v55);
              v44 = (__m128i *)*((_QWORD *)v42 + 6);
            }
            *((_QWORD *)v42 + 6) = v44 + 1;
          }
          goto LABEL_12;
        }
      }
      else
      {
        v42 = &v61;
      }
      v55.m128i_i64[0] = (__int64)&v51;
      v42 = (int *)sub_1239060(&v60, (__int64)v42, (unsigned int **)&v55);
      goto LABEL_66;
    }
LABEL_12:
    v10 = v54.m128i_i64[0];
    v11 = (16 * v52) | v50 & 7 | (8 * (unsigned int)((_DWORD)v53 != 0));
    v12 = *(unsigned int *)(a2 + 8);
    if ( v12 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v47 = v54.m128i_i64[0];
      sub_C8D5F0(a2, (const void *)(a2 + 16), v12 + 1, 0x10u, v8, v54.m128i_i64[0]);
      v12 = *(unsigned int *)(a2 + 8);
      v10 = v47;
    }
    v13 = (__int64 *)(*(_QWORD *)a2 + 16 * v12);
    v14 = a1;
    *v13 = v10;
    v15 = 13;
    v13[1] = v11;
    ++*(_DWORD *)(a2 + 8);
    if ( (unsigned __int8)sub_120AFE0(a1, 13, "expected ')' in call") )
      goto LABEL_56;
    if ( *(_DWORD *)(a1 + 240) != 4 )
      break;
  }
  v17 = a1;
  v48 = a1 + 1536;
  if ( v63 == &v61 )
    goto LABEL_46;
  v46 = a1;
  v18 = (__int64)v63;
  while ( 2 )
  {
    v55.m128i_i32[0] = *(_DWORD *)(v18 + 32);
    v19 = *(const __m128i **)(v18 + 48);
    v20 = *(const __m128i **)(v18 + 40);
    v56 = 0;
    v57 = 0;
    v55.m128i_i64[1] = 0;
    v21 = (char *)v19 - (char *)v20;
    if ( v19 == v20 )
    {
      v23 = 0;
    }
    else
    {
      if ( v21 > 0x7FFFFFFFFFFFFFF0LL )
        sub_4261EA(v14, v15, v16);
      v22 = sub_22077B0(v21);
      v19 = *(const __m128i **)(v18 + 48);
      v23 = v22;
      v20 = *(const __m128i **)(v18 + 40);
    }
    v55.m128i_i64[1] = v23;
    v56 = v23;
    v57 = v23 + v21;
    if ( v19 == v20 )
    {
      v25 = (__m128i *)v23;
    }
    else
    {
      v24 = (__m128i *)v23;
      v25 = (__m128i *)(v23 + (char *)v19 - (char *)v20);
      do
      {
        if ( v24 )
          *v24 = _mm_loadu_si128(v20);
        ++v24;
        ++v20;
      }
      while ( v24 != v25 );
    }
    v56 = (__int64)v25;
    v26 = *(_QWORD *)(v46 + 1544);
    if ( !v26 )
    {
      v27 = v48;
      goto LABEL_33;
    }
    v15 = v55.m128i_u32[0];
    v27 = v48;
    do
    {
      while ( 1 )
      {
        v28 = *(_QWORD *)(v26 + 16);
        v29 = *(_QWORD *)(v26 + 24);
        if ( *(_DWORD *)(v26 + 32) >= v55.m128i_i32[0] )
          break;
        v26 = *(_QWORD *)(v26 + 24);
        if ( !v29 )
          goto LABEL_31;
      }
      v27 = v26;
      v26 = *(_QWORD *)(v26 + 16);
    }
    while ( v28 );
LABEL_31:
    if ( v27 == v48 || v55.m128i_i32[0] < *(_DWORD *)(v27 + 32) )
    {
LABEL_33:
      v15 = v27;
      v54.m128i_i64[0] = (__int64)&v55;
      v30 = sub_12395C0((_QWORD *)(v46 + 1528), v27, (unsigned int **)&v54);
      v25 = (__m128i *)v56;
      v23 = v55.m128i_i64[1];
      v27 = v30;
    }
    if ( v25 != (__m128i *)v23 )
    {
      v31 = (const __m128i *)v23;
      v32 = &v54.m128i_i64[1];
      v33 = v25;
      do
      {
        while ( 1 )
        {
          v34 = *(_QWORD *)a2 + 16LL * v31->m128i_u32[0];
          v54 = _mm_loadu_si128(v31);
          v53 = v34;
          v15 = *(_QWORD *)(v27 + 48);
          if ( v15 != *(_QWORD *)(v27 + 56) )
            break;
          ++v31;
          v49 = v32;
          sub_12135D0((const __m128i **)(v27 + 40), (const __m128i *)v15, &v53, v32);
          v32 = v49;
          if ( v33 == v31 )
            goto LABEL_41;
        }
        if ( v15 )
        {
          *(_QWORD *)v15 = v34;
          *(_QWORD *)(v15 + 8) = v54.m128i_i64[1];
          v15 = *(_QWORD *)(v27 + 48);
        }
        v15 += 16;
        ++v31;
        *(_QWORD *)(v27 + 48) = v15;
      }
      while ( v33 != v31 );
LABEL_41:
      v23 = v55.m128i_i64[1];
    }
    if ( v23 )
    {
      v15 = v57 - v23;
      j_j___libc_free_0(v23, v57 - v23);
    }
    v14 = v18;
    v18 = sub_220EEE0(v18);
    if ( (int *)v18 != &v61 )
      continue;
    break;
  }
  v17 = v46;
LABEL_46:
  v3 = sub_120AFE0(v17, 13, "expected ')' in calls");
LABEL_47:
  sub_1207E40(v62);
  return v3;
}
