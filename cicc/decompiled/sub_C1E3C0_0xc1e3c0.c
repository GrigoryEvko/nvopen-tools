// Function: sub_C1E3C0
// Address: 0xc1e3c0
//
__int64 __fastcall sub_C1E3C0(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 *v4; // rbx
  unsigned int v5; // r13d
  size_t v6; // r14
  const void *v7; // r8
  __int64 v8; // r9
  size_t v9; // rdx
  unsigned int v10; // r13d
  int v11; // r11d
  unsigned int v12; // r14d
  bool v13; // r10
  __int64 v14; // rcx
  unsigned __int64 *v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rsi
  __int64 v18; // r14
  int v19; // esi
  int v20; // eax
  __int64 result; // rax
  const __m128i *v22; // r15
  int v23; // eax
  unsigned int v24; // r14d
  unsigned int v25; // esi
  int v26; // eax
  unsigned __int64 *v27; // rdx
  int v28; // eax
  unsigned int v29; // esi
  int v30; // eax
  unsigned __int64 *v31; // rdx
  int v32; // eax
  int v33; // eax
  _QWORD *v35; // [rsp+18h] [rbp-128h]
  __int64 v36; // [rsp+20h] [rbp-120h]
  int v37; // [rsp+2Ch] [rbp-114h]
  __int64 v38; // [rsp+30h] [rbp-110h]
  __int64 v39; // [rsp+38h] [rbp-108h]
  __int64 v40; // [rsp+40h] [rbp-100h]
  size_t v41; // [rsp+40h] [rbp-100h]
  __int64 v42; // [rsp+48h] [rbp-F8h]
  __int64 v43; // [rsp+48h] [rbp-F8h]
  const void *v44; // [rsp+48h] [rbp-F8h]
  unsigned __int64 *v45; // [rsp+58h] [rbp-E8h] BYREF
  unsigned __int64 *v46[2]; // [rsp+60h] [rbp-E0h] BYREF
  __m128i v47[13]; // [rsp+70h] [rbp-D0h] BYREF

  v3 = a1[2];
  v47[0].m128i_i64[1] = a1[3];
  v47[0].m128i_i64[0] = v3;
  if ( (unsigned __int8)sub_C1C670(a2, (__int64)v47, &v45) )
    goto LABEL_2;
  v29 = *(_DWORD *)(a2 + 24);
  v30 = *(_DWORD *)(a2 + 16);
  v31 = v45;
  ++*(_QWORD *)a2;
  v32 = v30 + 1;
  v46[0] = v31;
  if ( 4 * v32 >= 3 * v29 )
  {
    v29 *= 2;
  }
  else if ( v29 - *(_DWORD *)(a2 + 20) - v32 > v29 >> 3 )
  {
    goto LABEL_44;
  }
  sub_C1E220(a2, v29);
  sub_C1C670(a2, (__int64)v47, v46);
  v31 = v46[0];
  v32 = *(_DWORD *)(a2 + 16) + 1;
LABEL_44:
  *(_DWORD *)(a2 + 16) = v32;
  if ( v31[1] != -1 || *v31 )
    --*(_DWORD *)(a2 + 20);
  *(__m128i *)v31 = _mm_loadu_si128(v47);
LABEL_2:
  v35 = a1 + 10;
  v39 = a1[12];
  if ( (_QWORD *)v39 == a1 + 10 )
    goto LABEL_23;
  do
  {
    v4 = *(__int64 **)(v39 + 64);
    if ( !v4 )
      goto LABEL_22;
    do
    {
      v5 = *(_DWORD *)(a2 + 24);
      if ( !v5 )
      {
        ++*(_QWORD *)a2;
        v18 = (__int64)(v4 + 1);
        v47[0].m128i_i64[0] = 0;
LABEL_15:
        v19 = 2 * v5;
        goto LABEL_16;
      }
      v6 = v4[2];
      v7 = (const void *)v4[1];
      v8 = *(_QWORD *)(a2 + 8);
      v9 = v6;
      if ( v7 )
      {
        v42 = *(_QWORD *)(a2 + 8);
        v40 = v4[1];
        sub_C7D030(v47);
        sub_C7D280(v47, v40, v6);
        sub_C7D290(v47, v46);
        LODWORD(v6) = v46[0];
        v7 = (const void *)v4[1];
        v9 = v4[2];
        v8 = v42;
      }
      v10 = v5 - 1;
      v11 = 1;
      v12 = v10 & v6;
      v13 = v7 == 0;
      v14 = 0;
      while ( 1 )
      {
        v15 = (unsigned __int64 *)(v8 + 16LL * v12);
        v16 = v15[1];
        v17 = *v15;
        if ( v16 != v9 )
          break;
        if ( (const void *)v17 == v7 )
          goto LABEL_21;
        if ( !v17 || v13 )
          break;
        v36 = v14;
        v37 = v11;
        v38 = v8;
        v41 = v9;
        v44 = v7;
        v23 = memcmp(v7, (const void *)v17, v9);
        v7 = v44;
        v9 = v41;
        v8 = v38;
        v11 = v37;
        v14 = v36;
        v13 = 0;
        if ( !v23 )
          goto LABEL_21;
LABEL_33:
        v24 = v11 + v12;
        ++v11;
        v12 = v10 & v24;
      }
      if ( v16 != -1 )
      {
        if ( !(v14 | v17) && v16 == -2 )
          v14 = v8 + 16LL * v12;
        goto LABEL_33;
      }
      if ( v17 )
        goto LABEL_33;
      v5 = *(_DWORD *)(a2 + 24);
      if ( !v14 )
        v14 = v8 + 16LL * v12;
      v33 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v18 = (__int64)(v4 + 1);
      v47[0].m128i_i64[0] = v14;
      v20 = v33 + 1;
      if ( 4 * v20 >= 3 * v5 )
        goto LABEL_15;
      if ( v5 - (v20 + *(_DWORD *)(a2 + 20)) <= v5 >> 3 )
      {
        v19 = v5;
LABEL_16:
        sub_C1E220(a2, v19);
        sub_C1C670(a2, v18, (unsigned __int64 **)v47);
        v14 = v47[0].m128i_i64[0];
        v20 = *(_DWORD *)(a2 + 16) + 1;
      }
      *(_DWORD *)(a2 + 16) = v20;
      if ( *(_QWORD *)v14 || *(_QWORD *)(v14 + 8) != -1 )
        --*(_DWORD *)(a2 + 20);
      *(__m128i *)v14 = _mm_loadu_si128((const __m128i *)(v4 + 1));
LABEL_21:
      v4 = (__int64 *)*v4;
    }
    while ( v4 );
LABEL_22:
    v39 = sub_220EF30(v39);
  }
  while ( v35 != (_QWORD *)v39 );
LABEL_23:
  result = (__int64)(a1 + 16);
  v43 = a1[18];
  if ( a1 + 16 != (_QWORD *)v43 )
  {
    while ( 1 )
    {
      v22 = *(const __m128i **)(v43 + 64);
      if ( (const __m128i *)(v43 + 48) != v22 )
        break;
LABEL_27:
      result = sub_220EF30(v43);
      v43 = result;
      if ( a1 + 16 == (_QWORD *)result )
        return result;
    }
    while ( 2 )
    {
      if ( (unsigned __int8)sub_C1C670(a2, (__int64)v22[2].m128i_i64, v46) )
      {
LABEL_26:
        sub_C1E3C0(&v22[3], a2);
        v22 = (const __m128i *)sub_220EF30(v22);
        if ( (const __m128i *)(v43 + 48) == v22 )
          goto LABEL_27;
        continue;
      }
      break;
    }
    v25 = *(_DWORD *)(a2 + 24);
    v26 = *(_DWORD *)(a2 + 16);
    v27 = v46[0];
    ++*(_QWORD *)a2;
    v28 = v26 + 1;
    v47[0].m128i_i64[0] = (__int64)v27;
    if ( 4 * v28 >= 3 * v25 )
    {
      v25 *= 2;
    }
    else if ( v25 - *(_DWORD *)(a2 + 20) - v28 > v25 >> 3 )
    {
      goto LABEL_36;
    }
    sub_C1E220(a2, v25);
    sub_C1C670(a2, (__int64)v22[2].m128i_i64, (unsigned __int64 **)v47);
    v27 = (unsigned __int64 *)v47[0].m128i_i64[0];
    v28 = *(_DWORD *)(a2 + 16) + 1;
LABEL_36:
    *(_DWORD *)(a2 + 16) = v28;
    if ( *v27 || v27[1] != -1 )
      --*(_DWORD *)(a2 + 20);
    *(__m128i *)v27 = _mm_loadu_si128(v22 + 2);
    goto LABEL_26;
  }
  return result;
}
