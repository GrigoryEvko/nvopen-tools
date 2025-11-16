// Function: sub_3235E40
// Address: 0x3235e40
//
__int64 __fastcall sub_3235E40(__int64 a1, const __m128i *a2)
{
  unsigned int v4; // r13d
  _QWORD *v5; // r14
  size_t v6; // r15
  __int64 v7; // r8
  __int64 i; // r9
  __int64 v9; // rcx
  __int64 v10; // r13
  int v11; // eax
  int v12; // r11d
  __int64 v13; // r10
  const void *v14; // rsi
  bool v15; // al
  int v16; // r9d
  unsigned int v17; // r13d
  int v18; // eax
  int v19; // r11d
  __int64 v20; // r10
  bool v21; // al
  const void *v22; // rsi
  int v23; // r9d
  int v24; // eax
  __int64 v25; // rax
  int v27; // eax
  int v28; // eax
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned int v31; // edi
  __int64 v32; // rdx
  __m128i *v33; // rsi
  __m128i *v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // r8
  __int64 v37; // rax
  __int64 v38; // rax
  int v39; // eax
  __int64 v40; // r13
  int v41; // eax
  int v42; // r11d
  const void *v43; // rsi
  bool v44; // al
  int v45; // r9d
  int v46; // eax
  unsigned __int64 v47; // r13
  __int64 v48; // rdi
  __int64 v49; // [rsp+0h] [rbp-C0h]
  __int64 v50; // [rsp+0h] [rbp-C0h]
  __int64 v51; // [rsp+0h] [rbp-C0h]
  __int64 v52; // [rsp+10h] [rbp-B0h]
  int v53; // [rsp+10h] [rbp-B0h]
  int v54; // [rsp+10h] [rbp-B0h]
  int v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+18h] [rbp-A8h]
  __int64 v57; // [rsp+18h] [rbp-A8h]
  unsigned int v58; // [rsp+24h] [rbp-9Ch]
  unsigned int v59; // [rsp+24h] [rbp-9Ch]
  unsigned int v60; // [rsp+24h] [rbp-9Ch]
  int v61; // [rsp+28h] [rbp-98h]
  __int64 v62; // [rsp+28h] [rbp-98h]
  unsigned int v63; // [rsp+28h] [rbp-98h]
  __int64 v64; // [rsp+28h] [rbp-98h]
  int v65; // [rsp+28h] [rbp-98h]
  unsigned int v66; // [rsp+28h] [rbp-98h]
  __int64 v67; // [rsp+28h] [rbp-98h]
  __m128i v68; // [rsp+50h] [rbp-70h] BYREF
  __int64 v69; // [rsp+60h] [rbp-60h]
  int v70; // [rsp+68h] [rbp-58h]
  unsigned __int64 v71; // [rsp+70h] [rbp-50h]
  __int64 v72; // [rsp+78h] [rbp-48h]
  __int64 v73; // [rsp+80h] [rbp-40h]
  __int64 v74; // [rsp+88h] [rbp-38h]

  v4 = *(_DWORD *)(a1 + 24);
  v5 = (_QWORD *)a2->m128i_i64[0];
  v6 = a2->m128i_u64[1];
  if ( !v4 )
  {
    ++*(_QWORD *)a1;
LABEL_3:
    sub_1253750(a1, 2 * v4);
    v9 = 0;
    v61 = *(_DWORD *)(a1 + 24);
    if ( !v61 )
      goto LABEL_25;
    v10 = *(_QWORD *)(a1 + 8);
    v11 = sub_C94890(v5, v6);
    v12 = 1;
    v13 = 0;
    v7 = (unsigned int)(v61 - 1);
    for ( i = (unsigned int)v7 & v11; ; i = (unsigned int)v7 & v16 )
    {
      v9 = v10 + 24LL * (unsigned int)i;
      v14 = *(const void **)v9;
      if ( *(_QWORD *)v9 == -1 )
        goto LABEL_59;
      v15 = (_QWORD *)((char *)v5 + 2) == 0;
      if ( v14 != (const void *)-2LL )
      {
        if ( v6 != *(_QWORD *)(v9 + 8) )
          goto LABEL_8;
        v53 = v12;
        v56 = v13;
        v59 = i;
        v63 = v7;
        if ( !v6 )
          goto LABEL_25;
        v50 = v10 + 24LL * (unsigned int)i;
        v27 = memcmp(v5, v14, v6);
        v9 = v50;
        v7 = v63;
        i = v59;
        v13 = v56;
        v12 = v53;
        v15 = v27 == 0;
      }
      if ( v15 )
        goto LABEL_25;
      if ( v14 == (const void *)-2LL && !v13 )
        v13 = v9;
LABEL_8:
      v16 = v12 + i;
      ++v12;
    }
  }
  v17 = v4 - 1;
  v62 = *(_QWORD *)(a1 + 8);
  v18 = sub_C94890(v5, a2->m128i_i64[1]);
  v19 = 1;
  v9 = 0;
  for ( i = v17 & v18; ; i = v17 & v23 )
  {
    v20 = v62 + 24LL * (unsigned int)i;
    v21 = (_QWORD *)((char *)v5 + 1) == 0;
    v22 = *(const void **)v20;
    if ( *(_QWORD *)v20 != -1 )
    {
      v21 = (_QWORD *)((char *)v5 + 2) == 0;
      if ( v22 != (const void *)-2LL )
      {
        if ( *(_QWORD *)(v20 + 8) != v6 )
          goto LABEL_13;
        v52 = v9;
        v55 = v19;
        v58 = i;
        if ( !v6 )
          goto LABEL_20;
        v49 = v62 + 24LL * (unsigned int)i;
        v24 = memcmp(v5, v22, v6);
        v20 = v49;
        i = v58;
        v19 = v55;
        v9 = v52;
        v21 = v24 == 0;
      }
    }
    if ( v21 )
    {
LABEL_20:
      v25 = *(unsigned int *)(v20 + 16);
      return *(_QWORD *)(a1 + 32) + (v25 << 6) + 16;
    }
    if ( v22 == (const void *)-1LL )
      break;
LABEL_13:
    if ( !v9 && v22 == (const void *)-2LL )
      v9 = v20;
    v23 = v19 + i;
    ++v19;
  }
  v39 = *(_DWORD *)(a1 + 16);
  v4 = *(_DWORD *)(a1 + 24);
  if ( !v9 )
    v9 = v20;
  ++*(_QWORD *)a1;
  v28 = v39 + 1;
  if ( 4 * v28 >= 3 * v4 )
    goto LABEL_3;
  if ( v4 - (v28 + *(_DWORD *)(a1 + 20)) > v4 >> 3 )
    goto LABEL_26;
  sub_1253750(a1, v4);
  v9 = 0;
  v65 = *(_DWORD *)(a1 + 24);
  if ( !v65 )
    goto LABEL_25;
  v40 = *(_QWORD *)(a1 + 8);
  v41 = sub_C94890(v5, v6);
  v42 = 1;
  v13 = 0;
  v7 = (unsigned int)(v65 - 1);
  i = (unsigned int)v7 & v41;
  while ( 2 )
  {
    v9 = v40 + 24LL * (unsigned int)i;
    v43 = *(const void **)v9;
    if ( *(_QWORD *)v9 != -1 )
    {
      v44 = (_QWORD *)((char *)v5 + 2) == 0;
      if ( v43 != (const void *)-2LL )
      {
        if ( v6 != *(_QWORD *)(v9 + 8) )
        {
LABEL_45:
          if ( v43 != (const void *)-2LL || v13 )
            v9 = v13;
          v45 = v42 + i;
          v13 = v9;
          ++v42;
          i = (unsigned int)v7 & v45;
          continue;
        }
        v54 = v42;
        v57 = v13;
        v60 = i;
        v66 = v7;
        if ( !v6 )
          goto LABEL_25;
        v51 = v40 + 24LL * (unsigned int)i;
        v46 = memcmp(v5, v43, v6);
        v9 = v51;
        v7 = v66;
        i = v60;
        v13 = v57;
        v42 = v54;
        v44 = v46 == 0;
      }
      if ( v44 )
        goto LABEL_25;
      if ( v43 == (const void *)-1LL )
        goto LABEL_52;
      goto LABEL_45;
    }
    break;
  }
LABEL_59:
  if ( v5 == (_QWORD *)-1LL )
    goto LABEL_25;
LABEL_52:
  if ( v13 )
    v9 = v13;
LABEL_25:
  v28 = *(_DWORD *)(a1 + 16) + 1;
LABEL_26:
  *(_DWORD *)(a1 + 16) = v28;
  if ( *(_QWORD *)v9 != -1 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v5;
  *(_QWORD *)(v9 + 8) = v6;
  *(_DWORD *)(v9 + 16) = 0;
  v25 = *(unsigned int *)(a1 + 40);
  v29 = *(unsigned int *)(a1 + 44);
  v69 = 0;
  v30 = v25 + 1;
  v70 = 0;
  v31 = v25;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v68 = _mm_loadu_si128(a2);
  if ( v25 + 1 > v29 )
  {
    v47 = *(_QWORD *)(a1 + 32);
    v67 = v9;
    v48 = a1 + 32;
    if ( v47 > (unsigned __int64)&v68 || (unsigned __int64)&v68 >= v47 + (v25 << 6) )
    {
      sub_3226A20(v48, v30, v29, v9, v7, i);
      v25 = *(unsigned int *)(a1 + 40);
      v32 = *(_QWORD *)(a1 + 32);
      v33 = &v68;
      v9 = v67;
      v31 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_3226A20(v48, v30, v29, v9, v7, i);
      v32 = *(_QWORD *)(a1 + 32);
      v25 = *(unsigned int *)(a1 + 40);
      v9 = v67;
      v33 = (__m128i *)((char *)&v68 + v32 - v47);
      v31 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v32 = *(_QWORD *)(a1 + 32);
    v33 = &v68;
  }
  v34 = (__m128i *)((v25 << 6) + v32);
  if ( v34 )
  {
    *v34 = _mm_loadu_si128(v33);
    v34[1].m128i_i64[0] = v33[1].m128i_i64[0];
    v34[1].m128i_i32[2] = v33[1].m128i_i32[2];
    v35 = v33[2].m128i_i64[0];
    v33[2].m128i_i64[0] = 0;
    v36 = v71;
    v34[2].m128i_i64[0] = v35;
    v37 = v33[2].m128i_i64[1];
    v33[2].m128i_i64[1] = 0;
    v34[2].m128i_i64[1] = v37;
    v38 = v33[3].m128i_i64[0];
    v33[3].m128i_i64[0] = 0;
    v34[3].m128i_i64[0] = v38;
    v34[3].m128i_i64[1] = v33[3].m128i_i64[1];
    v31 = *(_DWORD *)(a1 + 40);
    *(_DWORD *)(a1 + 40) = v31 + 1;
    v25 = v31;
    if ( v36 )
    {
      v64 = v9;
      j_j___libc_free_0(v36);
      v9 = v64;
      v25 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
      v31 = *(_DWORD *)(a1 + 40) - 1;
    }
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v31 + 1;
  }
  *(_DWORD *)(v9 + 16) = v31;
  return *(_QWORD *)(a1 + 32) + (v25 << 6) + 16;
}
