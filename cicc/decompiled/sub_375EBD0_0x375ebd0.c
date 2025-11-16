// Function: sub_375EBD0
// Address: 0x375ebd0
//
__int64 *__fastcall sub_375EBD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // rbx
  __m128i *v9; // r12
  __int64 v10; // rax
  __int64 v11; // r14
  __int32 v12; // r13d
  unsigned __int64 v13; // rdi
  const __m128i *v14; // rdx
  unsigned __int64 v15; // rsi
  __int64 *v16; // r12
  int v17; // eax
  __m128i *v19; // r11
  __m128i *v20; // r14
  unsigned __int64 v21; // rcx
  __m128i *v22; // r10
  __m128i *v23; // rsi
  const __m128i *v24; // rax
  __m128i *v25; // r10
  const __m128i *v26; // rax
  __m128i *v27; // rax
  const __m128i *v28; // r9
  int v29; // eax
  __int64 v30; // rax
  const __m128i *v31; // [rsp+0h] [rbp-A0h]
  __int8 *v32; // [rsp+8h] [rbp-98h]
  unsigned __int64 v33; // [rsp+10h] [rbp-90h]
  unsigned __int64 v34; // [rsp+10h] [rbp-90h]
  __int64 v35; // [rsp+20h] [rbp-80h]
  int v36; // [rsp+2Ch] [rbp-74h]
  __m128i v39; // [rsp+40h] [rbp-60h] BYREF
  __m128i *v40; // [rsp+50h] [rbp-50h] BYREF
  __m128i *v41; // [rsp+58h] [rbp-48h]
  __m128i *v42; // [rsp+60h] [rbp-40h]

  v6 = a1;
  v7 = *(unsigned int *)(a2 + 64);
  v40 = 0;
  v41 = 0;
  v42 = 0;
  if ( !(_DWORD)v7 )
  {
    *(_DWORD *)(a2 + 36) = 0;
    v16 = (__int64 *)a2;
    goto LABEL_50;
  }
  v8 = 0;
  v36 = 0;
  v35 = 40 * v7;
  do
  {
    while ( 1 )
    {
      v10 = v8 + *(_QWORD *)(a2 + 40);
      v11 = *(_QWORD *)v10;
      v12 = *(_DWORD *)(v10 + 8);
      v39.m128i_i64[0] = *(_QWORD *)v10;
      v39.m128i_i32[2] = v12;
      sub_375EAB0(a1, (__int64)&v39);
      v9 = v41;
      v13 = (unsigned __int64)v40;
      v36 += *(_DWORD *)(v39.m128i_i64[0] + 36) == -3;
      if ( v41 != v40 )
        break;
      if ( v11 == v39.m128i_i64[0] && v12 == v39.m128i_i32[2] )
        goto LABEL_7;
      a6 = (__int64)v42;
      v14 = *(const __m128i **)(a2 + 40);
      a5 = (__int64)&v14->m128i_i64[(unsigned __int64)v8 / 8];
      if ( v14 != (const __m128i *)&v14->m128i_i8[v8] )
      {
        v15 = 0xCCCCCCCCCCCCCCCDLL * (v8 >> 3);
        if ( v15 > v42 - v41 )
        {
          if ( v15 )
          {
            v31 = *(const __m128i **)(a2 + 40);
            v32 = &v14->m128i_i8[v8];
            v34 = 0xCCCCCCCCCCCCCCDLL * (v8 >> 3);
            v27 = (__m128i *)sub_22077B0(v34 * 16);
            v13 = (unsigned __int64)v40;
            a5 = (__int64)v32;
            v14 = v31;
            v21 = (unsigned __int64)v27;
            if ( v9 == v40 )
            {
              v19 = v41;
              v22 = v27;
              v20 = &v27[v34];
            }
            else
            {
              v28 = v40;
              v22 = (__m128i *)((char *)v27 + (char *)v9 - (char *)v40);
              do
              {
                if ( v27 )
                  *v27 = _mm_loadu_si128(v28);
                ++v27;
                ++v28;
              }
              while ( v27 != v22 );
              v19 = v41;
              v20 = (__m128i *)(v21 - 0x3333333333333330LL * (v8 >> 3));
            }
          }
          else
          {
            v19 = v41;
            v20 = 0;
            v21 = 0;
            v22 = 0;
          }
          v23 = v22;
          v24 = v14;
          do
          {
            if ( v23 )
              *v23 = _mm_loadu_si128(v24);
            v24 = (const __m128i *)((char *)v24 + 40);
            ++v23;
          }
          while ( (const __m128i *)a5 != v24 );
          a5 -= (__int64)v14;
          v25 = &v22[((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)(a5 - 40) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1];
          if ( v9 == v19 )
          {
            v9 = v25;
          }
          else
          {
            v26 = v9;
            v9 = (__m128i *)((char *)v25 + (char *)v19 - (char *)v9);
            do
            {
              if ( v25 )
                *v25 = _mm_loadu_si128(v26);
              ++v25;
              ++v26;
            }
            while ( v9 != v25 );
          }
          if ( v13 )
          {
            v33 = v21;
            j_j___libc_free_0(v13);
            v21 = v33;
          }
          v40 = (__m128i *)v21;
          a6 = (__int64)v20;
          v41 = v9;
          v42 = v20;
        }
        else
        {
          do
          {
            if ( v9 )
              *v9 = _mm_loadu_si128(v14);
            v14 = (const __m128i *)((char *)v14 + 40);
            ++v9;
          }
          while ( (const __m128i *)a5 != v14 );
          a6 = (__int64)v42;
          v41 -= 0x333333333333333LL * (v8 >> 3);
          v9 = v41;
        }
      }
      if ( v9 == (__m128i *)a6 )
        goto LABEL_19;
      if ( v9 )
      {
        *v9 = _mm_loadu_si128(&v39);
        v9 = v41;
      }
LABEL_6:
      v41 = v9 + 1;
LABEL_7:
      v8 += 40;
      if ( v35 == v8 )
        goto LABEL_20;
    }
    if ( v41 != v42 )
    {
      if ( v41 )
      {
        *v41 = _mm_loadu_si128(&v39);
        v9 = v41;
      }
      goto LABEL_6;
    }
LABEL_19:
    v8 += 40;
    sub_33764F0((unsigned __int64 *)&v40, v9, &v39);
  }
  while ( v35 != v8 );
LABEL_20:
  v6 = a1;
  if ( v41 == v40 )
  {
    v16 = (__int64 *)a2;
    v29 = *(_DWORD *)(a2 + 64) - v36;
    *(_DWORD *)(a2 + 36) = v29;
    if ( !v29 )
      goto LABEL_50;
  }
  else
  {
    v16 = sub_33EC210(*(_QWORD **)(a1 + 8), (__int64 *)a2, (__int64)v40, v41 - v40);
    if ( (__int64 *)a2 != v16 )
    {
      *(_DWORD *)(a2 + 36) = -1;
      if ( *((_DWORD *)v16 + 9) < 0xFFFFFFFE )
        goto LABEL_24;
    }
    v17 = *((_DWORD *)v16 + 16) - v36;
    *((_DWORD *)v16 + 9) = v17;
    if ( v17 )
      goto LABEL_24;
LABEL_50:
    v30 = *(unsigned int *)(v6 + 1616);
    if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 1620) )
    {
      sub_C8D5F0(v6 + 1608, (const void *)(v6 + 1624), v30 + 1, 8u, a5, a6);
      v30 = *(unsigned int *)(v6 + 1616);
    }
    *(_QWORD *)(*(_QWORD *)(v6 + 1608) + 8 * v30) = v16;
    ++*(_DWORD *)(v6 + 1616);
  }
LABEL_24:
  if ( v40 )
    j_j___libc_free_0((unsigned __int64)v40);
  return v16;
}
