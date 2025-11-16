// Function: sub_E41FB0
// Address: 0xe41fb0
//
char __fastcall sub_E41FB0(__int64 a1, _QWORD *a2, size_t a3, __int8 *a4, size_t a5)
{
  __m128i *v9; // rcx
  size_t v11; // rax
  __m128i *v12; // rdx
  unsigned int v13; // r12d
  unsigned int v14; // r12d
  __int64 v15; // r13
  int v16; // eax
  int v17; // r11d
  __int64 v18; // r8
  unsigned int i; // r10d
  __int64 v20; // r9
  const void *v21; // rsi
  char result; // al
  unsigned int v23; // r10d
  __int64 v24; // rdi
  int v25; // r13d
  __int64 v26; // r12
  int v27; // r13d
  int v28; // eax
  int v29; // r11d
  __int64 v30; // r10
  unsigned int v31; // r9d
  const void *v32; // rsi
  bool v33; // al
  unsigned int v34; // r9d
  int v35; // eax
  __m128i *v36; // rdi
  int v37; // eax
  int v38; // eax
  __m128i *v39; // rax
  int v40; // eax
  int v41; // r13d
  __int64 v42; // r12
  int v43; // r13d
  int v44; // eax
  int v45; // r11d
  unsigned int v46; // r9d
  const void *v47; // rsi
  bool v48; // al
  unsigned int v49; // r9d
  int v50; // eax
  __m128i *v51; // [rsp+0h] [rbp-90h]
  __m128i *v52; // [rsp+0h] [rbp-90h]
  __m128i *v53; // [rsp+0h] [rbp-90h]
  __int64 v54; // [rsp+8h] [rbp-88h]
  __int64 v55; // [rsp+8h] [rbp-88h]
  __int64 v56; // [rsp+10h] [rbp-80h]
  int v57; // [rsp+1Ch] [rbp-74h]
  int v58; // [rsp+1Ch] [rbp-74h]
  int v59; // [rsp+1Ch] [rbp-74h]
  __int64 v60; // [rsp+20h] [rbp-70h]
  __int64 v61; // [rsp+20h] [rbp-70h]
  __int64 v62; // [rsp+20h] [rbp-70h]
  __m128i *v63; // [rsp+28h] [rbp-68h]
  unsigned int v64; // [rsp+28h] [rbp-68h]
  unsigned int v65; // [rsp+28h] [rbp-68h]
  __m128i *v66; // [rsp+28h] [rbp-68h]
  unsigned int v67; // [rsp+28h] [rbp-68h]
  __int64 v68; // [rsp+38h] [rbp-58h] BYREF
  __m128i *v69; // [rsp+40h] [rbp-50h] BYREF
  size_t v70; // [rsp+48h] [rbp-48h]
  __m128i v71[4]; // [rsp+50h] [rbp-40h] BYREF

  if ( !a4 )
  {
    v13 = *(_DWORD *)(a1 + 1464);
    v9 = v71;
    v70 = 0;
    v69 = v71;
    v71[0].m128i_i8[0] = 0;
    if ( v13 )
      goto LABEL_6;
LABEL_15:
    ++*(_QWORD *)(a1 + 1440);
    v24 = a1 + 1440;
    goto LABEL_16;
  }
  v9 = v71;
  v68 = a5;
  v69 = v71;
  v11 = a5;
  if ( a5 > 0xF )
  {
    v69 = (__m128i *)sub_22409D0(&v69, &v68, 0);
    v36 = v69;
    v71[0].m128i_i64[0] = v68;
LABEL_29:
    memcpy(v36, a4, a5);
    v11 = v68;
    v12 = v69;
    v9 = v71;
    goto LABEL_5;
  }
  if ( a5 == 1 )
  {
    v71[0].m128i_i8[0] = *a4;
    v12 = v71;
    goto LABEL_5;
  }
  if ( a5 )
  {
    v36 = v71;
    goto LABEL_29;
  }
  v12 = v71;
LABEL_5:
  v70 = v11;
  v12->m128i_i8[v11] = 0;
  v13 = *(_DWORD *)(a1 + 1464);
  if ( !v13 )
    goto LABEL_15;
LABEL_6:
  v14 = v13 - 1;
  v15 = *(_QWORD *)(a1 + 1448);
  v16 = sub_C94890(a2, a3);
  v9 = v71;
  v17 = 1;
  v18 = 0;
  for ( i = v14 & v16; ; i = v14 & v23 )
  {
    v20 = v15 + 48LL * i;
    v21 = *(const void **)v20;
    result = (_QWORD *)((char *)a2 + 1) == 0;
    if ( *(_QWORD *)v20 != -1 )
    {
      result = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v21 != (const void *)-2LL )
      {
        if ( *(_QWORD *)(v20 + 8) != a3 )
          goto LABEL_10;
        v56 = v18;
        v57 = v17;
        v60 = v15 + 48LL * i;
        v64 = i;
        if ( !a3 )
          goto LABEL_25;
        v51 = v9;
        v35 = memcmp(a2, v21, a3);
        v9 = v51;
        i = v64;
        v20 = v60;
        v17 = v57;
        v18 = v56;
        result = v35 == 0;
      }
    }
    if ( result )
    {
LABEL_25:
      if ( v69 != v9 )
        return j_j___libc_free_0(v69, v71[0].m128i_i64[0] + 1);
      return result;
    }
    if ( v21 == (const void *)-1LL )
      break;
LABEL_10:
    if ( v21 == (const void *)-2LL && !v18 )
      v18 = v20;
    v23 = v17 + i;
    ++v17;
  }
  v40 = *(_DWORD *)(a1 + 1456);
  v13 = *(_DWORD *)(a1 + 1464);
  v24 = a1 + 1440;
  if ( !v18 )
    v18 = v20;
  ++*(_QWORD *)(a1 + 1440);
  v38 = v40 + 1;
  if ( 4 * v38 < 3 * v13 )
  {
    if ( v13 - (v38 + *(_DWORD *)(a1 + 1460)) > v13 >> 3 )
      goto LABEL_36;
    v66 = v9;
    sub_E41C70(v24, v13);
    v41 = *(_DWORD *)(a1 + 1464);
    v18 = 0;
    v9 = v66;
    if ( !v41 )
      goto LABEL_35;
    v42 = *(_QWORD *)(a1 + 1448);
    v43 = v41 - 1;
    v44 = sub_C94890(a2, a3);
    v9 = v66;
    v45 = 1;
    v30 = 0;
    v46 = v43 & v44;
    while ( 2 )
    {
      v18 = v42 + 48LL * v46;
      v47 = *(const void **)v18;
      if ( *(_QWORD *)v18 == -1 )
        goto LABEL_62;
      v48 = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v47 != (const void *)-2LL )
      {
        if ( a3 != *(_QWORD *)(v18 + 8) )
        {
LABEL_52:
          if ( v30 || v47 != (const void *)-2LL )
            v18 = v30;
          v49 = v45 + v46;
          v30 = v18;
          ++v45;
          v46 = v43 & v49;
          continue;
        }
        v59 = v45;
        v62 = v30;
        v67 = v46;
        if ( !a3 )
          goto LABEL_35;
        v53 = v9;
        v55 = v42 + 48LL * v46;
        v50 = memcmp(a2, v47, a3);
        v18 = v55;
        v9 = v53;
        v46 = v67;
        v30 = v62;
        v45 = v59;
        v48 = v50 == 0;
      }
      break;
    }
    if ( v48 )
      goto LABEL_35;
    if ( v47 == (const void *)-1LL )
      goto LABEL_59;
    goto LABEL_52;
  }
LABEL_16:
  v63 = v9;
  sub_E41C70(v24, 2 * v13);
  v25 = *(_DWORD *)(a1 + 1464);
  v18 = 0;
  v9 = v63;
  if ( !v25 )
    goto LABEL_35;
  v26 = *(_QWORD *)(a1 + 1448);
  v27 = v25 - 1;
  v28 = sub_C94890(a2, a3);
  v9 = v63;
  v29 = 1;
  v30 = 0;
  v31 = v27 & v28;
  while ( 2 )
  {
    v18 = v26 + 48LL * v31;
    v32 = *(const void **)v18;
    if ( *(_QWORD *)v18 != -1 )
    {
      v33 = (_QWORD *)((char *)a2 + 2) == 0;
      if ( v32 != (const void *)-2LL )
      {
        if ( a3 != *(_QWORD *)(v18 + 8) )
        {
LABEL_21:
          v34 = v29 + v31;
          ++v29;
          v31 = v27 & v34;
          continue;
        }
        v58 = v29;
        v61 = v30;
        v65 = v31;
        if ( !a3 )
          goto LABEL_35;
        v52 = v9;
        v54 = v26 + 48LL * v31;
        v37 = memcmp(a2, v32, a3);
        v18 = v54;
        v9 = v52;
        v31 = v65;
        v30 = v61;
        v29 = v58;
        v33 = v37 == 0;
      }
      if ( v33 )
        goto LABEL_35;
      if ( v32 == (const void *)-2LL && !v30 )
        v30 = v18;
      goto LABEL_21;
    }
    break;
  }
LABEL_62:
  if ( a2 != (_QWORD *)-1LL )
  {
LABEL_59:
    if ( v30 )
      v18 = v30;
  }
LABEL_35:
  v38 = *(_DWORD *)(a1 + 1456) + 1;
LABEL_36:
  *(_DWORD *)(a1 + 1456) = v38;
  if ( *(_QWORD *)v18 != -1 )
    --*(_DWORD *)(a1 + 1460);
  *(_QWORD *)v18 = a2;
  *(_QWORD *)(v18 + 16) = v18 + 32;
  v39 = v69;
  *(_QWORD *)(v18 + 8) = a3;
  if ( v39 == v9 )
  {
    *(__m128i *)(v18 + 32) = _mm_load_si128(v71);
  }
  else
  {
    *(_QWORD *)(v18 + 16) = v39;
    *(_QWORD *)(v18 + 32) = v71[0].m128i_i64[0];
  }
  result = v70;
  *(_QWORD *)(v18 + 24) = v70;
  return result;
}
