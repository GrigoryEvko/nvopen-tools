// Function: sub_2F76DA0
// Address: 0x2f76da0
//
__int64 __fastcall sub_2F76DA0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7,
        __int64 a8,
        __int64 a9)
{
  __int64 v11; // rcx
  __m128i *v12; // r8
  unsigned int v13; // r12d
  __m128i *v14; // rsi
  signed __int64 m128i_i64; // rdx
  __m128i *v16; // rax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // r11
  __int64 v20; // r10
  char v21; // r9
  __int64 v22; // rdi
  __int64 (*v23)(); // rax
  const __m128i *v25; // rax
  char v26; // al
  __int64 v27; // r9
  __int64 v28; // [rsp+0h] [rbp-50h]
  __int64 v29; // [rsp+8h] [rbp-48h]
  __int64 v30; // [rsp+8h] [rbp-48h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  __int64 v33; // [rsp+10h] [rbp-40h]
  __int64 v34; // [rsp+18h] [rbp-38h]
  unsigned __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]

  v11 = *(unsigned int *)(a2 + 8);
  v12 = *(__m128i **)a2;
  v13 = a7;
  v14 = (__m128i *)(*(_QWORD *)a2 + 24 * v11);
  m128i_i64 = 0xAAAAAAAAAAAAAAABLL * ((24 * v11) >> 3);
  if ( !(m128i_i64 >> 2) )
  {
    v16 = v12;
LABEL_15:
    if ( m128i_i64 != 2 )
    {
      if ( m128i_i64 != 3 )
      {
        if ( m128i_i64 != 1 )
          goto LABEL_19;
        goto LABEL_18;
      }
      if ( a7 == v16->m128i_i32[0] )
        goto LABEL_8;
      v16 = (__m128i *)((char *)v16 + 24);
    }
    if ( a7 == v16->m128i_i32[0] )
      goto LABEL_8;
    v16 = (__m128i *)((char *)v16 + 24);
LABEL_18:
    if ( a7 != v16->m128i_i32[0] )
      goto LABEL_19;
    goto LABEL_8;
  }
  v16 = v12;
  m128i_i64 = (signed __int64)v12[6 * (m128i_i64 >> 2)].m128i_i64;
  while ( a7 != v16->m128i_i32[0] )
  {
    if ( a7 == v16[1].m128i_i32[2] )
    {
      v16 = (__m128i *)((char *)v16 + 24);
      break;
    }
    if ( a7 == v16[3].m128i_i32[0] )
    {
      v16 += 3;
      break;
    }
    if ( a7 == v16[4].m128i_i32[2] )
    {
      v16 = (__m128i *)((char *)v16 + 72);
      break;
    }
    v16 += 6;
    if ( v16 == (__m128i *)m128i_i64 )
    {
      m128i_i64 = 0xAAAAAAAAAAAAAAABLL * (((char *)v14 - (char *)v16) >> 3);
      goto LABEL_15;
    }
  }
LABEL_8:
  if ( v14 != v16 )
  {
    v17 = v16->m128i_i64[1];
    v18 = v16[1].m128i_i64[0];
    v19 = v17 | a8;
    v20 = v18 | a9;
    v16->m128i_i64[1] = v17 | a8;
    v16[1].m128i_i64[0] = v20;
    goto LABEL_10;
  }
LABEL_19:
  m128i_i64 = v11 + 1;
  v19 = a8;
  v25 = (const __m128i *)&a7;
  v20 = a9;
  if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
  {
    v27 = a2 + 16;
    if ( v12 > (__m128i *)&a7 || v14 <= (__m128i *)&a7 )
    {
      v33 = a9;
      v36 = a8;
      sub_C8D5F0(a2, (const void *)(a2 + 16), m128i_i64, 0x18u, (__int64)v12, v27);
      v19 = v36;
      v20 = v33;
      v25 = (const __m128i *)&a7;
      m128i_i64 = *(_QWORD *)a2;
      v14 = (__m128i *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
    }
    else
    {
      v30 = a9;
      v32 = a8;
      v35 = (unsigned __int64)v12;
      sub_C8D5F0(a2, (const void *)(a2 + 16), m128i_i64, 0x18u, (__int64)v12, v27);
      m128i_i64 = *(_QWORD *)a2;
      v20 = v30;
      v19 = v32;
      v25 = (const __m128i *)((char *)&a7 + *(_QWORD *)a2 - v35);
      v14 = (__m128i *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
    }
  }
  v18 = 0;
  v17 = 0;
  *v14 = _mm_loadu_si128(v25);
  v14[1].m128i_i64[0] = v25[1].m128i_i64[0];
  ++*(_DWORD *)(a2 + 8);
LABEL_10:
  v21 = 0;
  if ( *(_BYTE *)(a1 + 58) )
  {
    v22 = *(_QWORD *)(a1 + 8);
    v23 = *(__int64 (**)())(*(_QWORD *)v22 + 432LL);
    if ( v23 != sub_2F73F20 )
    {
      v28 = v20;
      v29 = v17;
      v31 = v19;
      v34 = v18;
      v26 = ((__int64 (__fastcall *)(__int64, __m128i *, signed __int64, __int64, __int64, _QWORD))v23)(
              v22,
              v14,
              m128i_i64,
              v17,
              v18,
              0);
      v20 = v28;
      v17 = v29;
      v19 = v31;
      v18 = v34;
      v21 = v26;
    }
  }
  return sub_2F74AE0(*(__int64 **)(a1 + 48), *(_QWORD **)(a1 + 24), v13, v17, v18, v21, v19, v20);
}
