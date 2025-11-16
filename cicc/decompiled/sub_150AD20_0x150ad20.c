// Function: sub_150AD20
// Address: 0x150ad20
//
const __m128i **__fastcall sub_150AD20(
        const __m128i **a1,
        __int64 a2,
        __int64 a3,
        unsigned __int64 a4,
        char a5,
        char a6,
        char a7)
{
  __int64 v10; // r15
  __m128i *v12; // rax
  const __m128i *v13; // rdi
  const __m128i *v14; // rsi
  int v15; // ecx
  __m128i *v16; // r8
  const __m128i *v17; // rdx
  __m128i *v18; // rsi
  unsigned int v19; // r15d
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rdx
  __m128i *v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  int v27; // [rsp+8h] [rbp-58h]
  unsigned int v28; // [rsp+8h] [rbp-58h]
  int v29; // [rsp+10h] [rbp-50h]
  __m128i *v30; // [rsp+10h] [rbp-50h]
  int v31; // [rsp+10h] [rbp-50h]
  __m128i v33; // [rsp+20h] [rbp-40h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a4 > 0x7FFFFFFFFFFFFFFLL )
    sub_4262D8((__int64)"vector::reserve");
  if ( a4 )
  {
    v29 = a4;
    v10 = a4;
    v12 = (__m128i *)sub_22077B0(16 * a4);
    v13 = *a1;
    v14 = a1[1];
    v15 = v29;
    v16 = v12;
    v17 = *a1;
    if ( v14 != *a1 )
    {
      v18 = (__m128i *)((char *)v12 + (char *)v14 - (char *)v13);
      do
      {
        if ( v12 )
          *v12 = _mm_loadu_si128(v17);
        ++v12;
        ++v17;
      }
      while ( v12 != v18 );
    }
    if ( v13 )
    {
      v27 = v29;
      v30 = v16;
      j_j___libc_free_0(v13, (char *)a1[2] - (char *)v13);
      v15 = v27;
      v16 = v30;
    }
    *a1 = v16;
    a1[1] = v16;
    a1[2] = &v16[v10];
    v31 = v15;
    if ( v15 )
    {
      v19 = 0;
      do
      {
        v24 = sub_150AAB0(a2, *(_QWORD *)(a3 + 8LL * v19));
        v20 = v19 + 1;
        v25 = v24;
        if ( a5 )
        {
          if ( !a6 )
            goto LABEL_24;
          v20 = v19 + 2;
          LODWORD(v21) = 0;
          LOBYTE(v22) = 0;
        }
        else
        {
          if ( !a6 )
          {
            if ( a7 )
            {
              LOBYTE(v22) = 0;
              v21 = *(_QWORD *)(a3 + 8LL * v20) & 0x1FFFFFFFLL;
              goto LABEL_14;
            }
            v20 = v19;
LABEL_24:
            LODWORD(v21) = 0;
            LOBYTE(v22) = 0;
            goto LABEL_14;
          }
          LODWORD(v21) = 0;
          v22 = *(_QWORD *)(a3 + 8LL * v20) & 7LL;
        }
LABEL_14:
        v33.m128i_i64[0] = v25;
        v23 = (__m128i *)a1[1];
        v33.m128i_i32[2] = (unsigned __int8)v22 | (8 * v21);
        if ( v23 == a1[2] )
        {
          v28 = v20;
          sub_142DD90(a1, v23, &v33);
          v20 = v28;
        }
        else
        {
          if ( v23 )
          {
            *v23 = _mm_loadu_si128(&v33);
            v23 = (__m128i *)a1[1];
          }
          a1[1] = v23 + 1;
        }
        v19 = v20 + 1;
      }
      while ( v31 != v20 + 1 );
    }
  }
  return a1;
}
