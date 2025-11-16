// Function: sub_2FF7BB0
// Address: 0x2ff7bb0
//
unsigned __int64 __fastcall sub_2FF7BB0(__m128i *a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  const __m128i *v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  unsigned __int64 v9; // r13
  unsigned __int64 result; // rax
  __int32 v11; // r12d
  __int64 v12; // rdx
  _DWORD *i; // rdx
  unsigned int v14; // r13d
  __int64 v15; // r11
  __int64 v16; // r10
  unsigned int v17; // r9d
  unsigned int *v18; // rdi
  __int64 v19; // r12
  unsigned int v20; // r8d
  unsigned int v21; // esi
  unsigned int j; // ecx
  unsigned int v23; // eax
  __int64 v24; // rcx
  unsigned int v25; // esi

  v3 = a2;
  a1[12].m128i_i64[0] = (__int64)a2;
  v4 = (const __m128i *)a2[25];
  *a1 = _mm_loadu_si128(v4);
  a1[1] = _mm_loadu_si128(v4 + 1);
  a1[2] = _mm_loadu_si128(v4 + 2);
  a1[3] = _mm_loadu_si128(v4 + 3);
  a1[4] = _mm_loadu_si128(v4 + 4);
  v5 = *(__int64 (**)())(*a2 + 128LL);
  v6 = 0;
  if ( v5 != sub_2DAC790 )
  {
    v6 = ((__int64 (__fastcall *)(_QWORD *))v5)(a2);
    v3 = (_QWORD *)a1[12].m128i_i64[0];
  }
  a1[12].m128i_i64[1] = v6;
  sub_EA11E0(v3, (__int64)a1[5].m128i_i64);
  v9 = a1[3].m128i_u32[0];
  result = a1[13].m128i_u32[2];
  v11 = a1[3].m128i_i32[0];
  if ( v9 != result )
  {
    if ( v9 >= result )
    {
      if ( v9 > a1[13].m128i_u32[3] )
      {
        sub_C8D5F0((__int64)a1[13].m128i_i64, &a1[14], a1[3].m128i_u32[0], 4u, v7, v8);
        result = a1[13].m128i_u32[2];
      }
      v12 = a1[13].m128i_i64[0];
      result = v12 + 4 * result;
      for ( i = (_DWORD *)(v12 + 4 * v9); i != (_DWORD *)result; result += 4LL )
      {
        if ( result )
          *(_DWORD *)result = 0;
      }
    }
    a1[13].m128i_i32[2] = v9;
  }
  v14 = a1->m128i_i32[0];
  a1[18].m128i_i32[1] = a1->m128i_i32[0];
  if ( v11 )
  {
    v15 = (unsigned int)(v11 - 1);
    v16 = a1[2].m128i_i64[0];
    v17 = v14;
    v18 = (unsigned int *)(v16 + 8);
    v19 = v16 + 32 * v15 + 40;
    do
    {
      while ( 1 )
      {
        v20 = *v18;
        if ( *v18 )
          break;
        v18 += 8;
        if ( (unsigned int *)v19 == v18 )
          goto LABEL_22;
      }
      if ( v17 )
      {
        v21 = *v18;
        for ( j = v17 % v20; j; j = v23 % j )
        {
          v23 = v21;
          v21 = j;
        }
        v17 = v20 * (v17 / v21);
      }
      v18 += 8;
      a1[18].m128i_i32[1] = v17;
    }
    while ( (unsigned int *)v19 != v18 );
LABEL_22:
    v24 = 0;
    a1[18].m128i_i32[0] = v17 / v14;
    while ( 1 )
    {
      v25 = *(_DWORD *)(v16 + 8 * v24 + 8);
      if ( v25 )
        v25 = a1[18].m128i_i32[1] / v25;
      result = a1[13].m128i_u64[0];
      *(_DWORD *)(result + v24) = v25;
      if ( 4 * v15 == v24 )
        break;
      v16 = a1[2].m128i_i64[0];
      v24 += 4;
    }
  }
  else
  {
    a1[18].m128i_i32[0] = 1;
  }
  return result;
}
