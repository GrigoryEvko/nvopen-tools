// Function: sub_2010420
// Address: 0x2010420
//
__int64 *__fastcall sub_2010420(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i *a5, const __m128i *a6)
{
  __int64 *v6; // r12
  __int64 v7; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rbx
  __int32 v13; // r12d
  __int64 v14; // rax
  bool v15; // zf
  const __m128i *v16; // rdi
  const __m128i *v17; // r12
  const __m128i *v18; // rsi
  unsigned __int64 v19; // rdx
  int v20; // eax
  __m128i *v22; // r11
  const __m128i *v23; // rcx
  __m128i *v24; // r10
  __m128i *v25; // rdx
  const __m128i *v26; // rax
  __m128i *v27; // r10
  const __m128i *v28; // rax
  __int64 *v29; // rax
  int v30; // eax
  __int64 v31; // rax
  __m128i *v32; // rax
  const __m128i *v33; // r9
  const __m128i *v34; // [rsp+0h] [rbp-90h]
  const __m128i *v35; // [rsp+0h] [rbp-90h]
  __m128i *v36; // [rsp+8h] [rbp-88h]
  __m128i *v37; // [rsp+10h] [rbp-80h]
  unsigned __int64 v38; // [rsp+10h] [rbp-80h]
  int v39; // [rsp+1Ch] [rbp-74h]
  __int64 v40; // [rsp+28h] [rbp-68h]
  __m128i v41; // [rsp+30h] [rbp-60h] BYREF
  const __m128i *v42; // [rsp+40h] [rbp-50h] BYREF
  __m128i *v43; // [rsp+48h] [rbp-48h]
  const __m128i *v44; // [rsp+50h] [rbp-40h]

  v6 = (__int64 *)a2;
  if ( *(_DWORD *)(a2 + 28) < 0xFFFFFFFE )
    return v6;
  v7 = *(unsigned int *)(a2 + 56);
  v42 = 0;
  v43 = 0;
  v44 = 0;
  if ( !(_DWORD)v7 )
  {
    *(_DWORD *)(a2 + 28) = 0;
    goto LABEL_50;
  }
  v39 = 0;
  v10 = 0;
  v40 = 40 * v7;
  do
  {
    v11 = v10 + *(_QWORD *)(a2 + 32);
    v12 = *(_QWORD *)v11;
    v13 = *(_DWORD *)(v11 + 8);
    v41.m128i_i64[0] = *(_QWORD *)v11;
    v41.m128i_i32[2] = v13;
    v14 = sub_2010420(a1, v41.m128i_i64[0]);
    v15 = *(_DWORD *)(v14 + 28) == -3;
    v41.m128i_i64[0] = v14;
    if ( v15 )
    {
      sub_2010110(a1, (__int64)&v41);
      v14 = v41.m128i_i64[0];
      if ( *(_DWORD *)(v41.m128i_i64[0] + 28) == -3 )
        ++v39;
    }
    a5 = v43;
    v16 = v42;
    if ( v42 == v43 )
    {
      if ( v12 != v14 || v13 != v41.m128i_i32[2] )
      {
        v17 = *(const __m128i **)(a2 + 32);
        a6 = v44;
        v18 = (const __m128i *)((char *)v17 + v10);
        if ( &v17->m128i_i8[v10] == (__int8 *)v17 )
        {
LABEL_18:
          if ( a6 != a5 )
            goto LABEL_19;
        }
        else
        {
          v19 = 0xCCCCCCCCCCCCCCCDLL * (v10 >> 3);
          if ( v19 <= v44 - v43 )
          {
            do
            {
              if ( a5 )
                *a5 = _mm_loadu_si128(v17);
              v17 = (const __m128i *)((char *)v17 + 40);
              ++a5;
            }
            while ( v18 != v17 );
            a6 = v44;
            v43 -= 0x333333333333333LL * (v10 >> 3);
            a5 = v43;
            goto LABEL_18;
          }
          if ( v19 )
          {
            v35 = v43;
            v38 = 0xCCCCCCCCCCCCCCDLL * (v10 >> 3);
            v32 = (__m128i *)sub_22077B0(v38 * 16);
            v16 = v42;
            a5 = (__m128i *)v35;
            v18 = (const __m128i *)((char *)v17 + v10);
            v23 = v32;
            if ( v35 == v42 )
            {
              v22 = v43;
              a6 = v44;
              v24 = v32;
              v37 = &v32[v38];
            }
            else
            {
              v33 = v42;
              v24 = (__m128i *)((char *)v32 + (char *)v35 - (char *)v42);
              do
              {
                if ( v32 )
                  *v32 = _mm_loadu_si128(v33);
                ++v32;
                ++v33;
              }
              while ( v24 != v32 );
              v22 = v43;
              a6 = v44;
              v37 = (__m128i *)&v23[0xFCCCCCCCCCCCCCCDLL * (v10 >> 3)];
            }
          }
          else
          {
            v37 = 0;
            v22 = v43;
            v23 = 0;
            v24 = 0;
          }
          v25 = v24;
          v26 = v17;
          do
          {
            if ( v25 )
              *v25 = _mm_loadu_si128(v26);
            v26 = (const __m128i *)((char *)v26 + 40);
            ++v25;
          }
          while ( v18 != v26 );
          v27 = &v24[((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)((char *)v18 - (char *)v17 - 40) >> 3))
                    & 0x1FFFFFFFFFFFFFFFLL)
                   + 1];
          if ( a5 == v22 )
          {
            a5 = v27;
          }
          else
          {
            v28 = a5;
            a5 = (__m128i *)((char *)v27 + (char *)v22 - (char *)a5);
            do
            {
              if ( v27 )
                *v27 = _mm_loadu_si128(v28);
              ++v27;
              ++v28;
            }
            while ( a5 != v27 );
          }
          if ( v16 )
          {
            v34 = v23;
            v36 = a5;
            j_j___libc_free_0(v16, (char *)a6 - (char *)v16);
            v23 = v34;
            a5 = v36;
          }
          v42 = v23;
          v43 = a5;
          a6 = v37;
          v44 = v37;
          if ( v37 != a5 )
          {
LABEL_19:
            if ( a5 )
            {
              *a5 = _mm_loadu_si128(&v41);
              a5 = v43;
            }
LABEL_7:
            v43 = ++a5;
            goto LABEL_8;
          }
        }
        sub_1D4B0A0(&v42, a6, &v41);
      }
    }
    else
    {
      if ( v43 != v44 )
      {
        if ( v43 )
        {
          *v43 = _mm_loadu_si128(&v41);
          a5 = v43;
        }
        goto LABEL_7;
      }
      sub_1D4B0A0(&v42, v43, &v41);
    }
LABEL_8:
    v10 += 40;
  }
  while ( v40 != v10 );
  if ( v43 == v42 )
  {
    v6 = (__int64 *)a2;
    v20 = *(_DWORD *)(a2 + 56) - v39;
    goto LABEL_25;
  }
  v29 = sub_1D2E160(*(_QWORD **)(a1 + 8), (__int64 *)a2, (__int64)v42, v43 - v42);
  v6 = v29;
  if ( v29 == (__int64 *)a2 )
  {
    v20 = *(_DWORD *)(a2 + 56) - v39;
LABEL_25:
    *((_DWORD *)v6 + 7) = v20;
    if ( !v20 )
    {
LABEL_50:
      v31 = *(unsigned int *)(a1 + 1376);
      if ( (unsigned int)v31 >= *(_DWORD *)(a1 + 1380) )
      {
        sub_16CD150(a1 + 1368, (const void *)(a1 + 1384), 0, 8, (int)a5, (int)a6);
        v31 = *(unsigned int *)(a1 + 1376);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 1368) + 8 * v31) = v6;
      ++*(_DWORD *)(a1 + 1376);
    }
  }
  else
  {
    *(_DWORD *)(a2 + 28) = -1;
    if ( *((_DWORD *)v29 + 7) >= 0xFFFFFFFE )
    {
      v30 = *((_DWORD *)v29 + 14) - v39;
      *((_DWORD *)v6 + 7) = v30;
      if ( !v30 )
        goto LABEL_50;
    }
  }
  if ( v42 )
    j_j___libc_free_0(v42, (char *)v44 - (char *)v42);
  return v6;
}
