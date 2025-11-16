// Function: sub_1F4B6B0
// Address: 0x1f4b6b0
//
unsigned __int64 __fastcall sub_1F4B6B0(__m128i *a1, _QWORD *a2)
{
  _QWORD *v3; // rdi
  const __m128i *v4; // rax
  __int64 (*v5)(); // rdx
  __int64 v6; // rax
  int v7; // r8d
  int v8; // r9d
  unsigned __int64 v9; // r13
  unsigned __int64 result; // rax
  unsigned __int32 v11; // r11d
  __int64 v12; // r8
  unsigned int *v13; // rsi
  __int64 v14; // r10
  unsigned __int64 v15; // rax
  unsigned int v16; // ecx
  unsigned __int64 v17; // rdi
  unsigned int v18; // edx
  __int64 v19; // rcx
  unsigned int v20; // esi
  __int64 v21; // rdx
  _DWORD *i; // rdx

  v3 = a2;
  a1[11].m128i_i64[0] = (__int64)a2;
  v4 = (const __m128i *)a2[20];
  *a1 = _mm_loadu_si128(v4);
  a1[1] = _mm_loadu_si128(v4 + 1);
  a1[2] = _mm_loadu_si128(v4 + 2);
  a1[3] = _mm_loadu_si128(v4 + 3);
  a1[4].m128i_i64[0] = v4[4].m128i_i64[0];
  v5 = *(__int64 (**)())(*a2 + 40LL);
  v6 = 0;
  if ( v5 != sub_1D00B00 )
  {
    v6 = ((__int64 (__fastcall *)(_QWORD *))v5)(a2);
    v3 = (_QWORD *)a1[11].m128i_i64[0];
  }
  a1[11].m128i_i64[1] = v6;
  sub_38E2200(v3, &a1[4].m128i_u64[1]);
  v9 = a1[3].m128i_u32[0];
  result = a1[12].m128i_u32[2];
  if ( v9 >= result )
  {
    if ( v9 <= result )
    {
      v11 = a1->m128i_i32[0];
      a1[17].m128i_i32[1] = a1->m128i_i32[0];
      if ( (_DWORD)v9 )
        goto LABEL_6;
LABEL_26:
      a1[17].m128i_i32[0] = 1;
      return result;
    }
    if ( v9 > a1[12].m128i_u32[3] )
    {
      sub_16CD150((__int64)a1[12].m128i_i64, &a1[13], a1[3].m128i_u32[0], 4, v7, v8);
      result = a1[12].m128i_u32[2];
    }
    v21 = a1[12].m128i_i64[0];
    result = v21 + 4 * result;
    for ( i = (_DWORD *)(v21 + 4 * v9); i != (_DWORD *)result; result += 4LL )
    {
      if ( result )
        *(_DWORD *)result = 0;
    }
  }
  v11 = a1->m128i_i32[0];
  a1[12].m128i_i32[2] = v9;
  a1[17].m128i_i32[1] = v11;
  if ( !(_DWORD)v9 )
    goto LABEL_26;
LABEL_6:
  v12 = a1[2].m128i_i64[0];
  v13 = (unsigned int *)(v12 + 8);
  v14 = v12 + 32LL * (unsigned int)(v9 - 1) + 40;
  LODWORD(v15) = v11;
  do
  {
    while ( 1 )
    {
      v16 = *v13;
      if ( *v13 )
        break;
      v13 += 8;
      if ( (unsigned int *)v14 == v13 )
        goto LABEL_13;
    }
    v17 = v16 * (unsigned __int64)(unsigned int)v15;
    while ( 1 )
    {
      v18 = (unsigned int)v15 % v16;
      LODWORD(v15) = v16;
      if ( !v18 )
        break;
      v16 = v18;
    }
    v13 += 8;
    v15 = v17 / v16;
    a1[17].m128i_i32[1] = v15;
  }
  while ( (unsigned int *)v14 != v13 );
LABEL_13:
  v19 = 0;
  a1[17].m128i_i32[0] = (unsigned int)v15 / v11;
  while ( 1 )
  {
    v20 = *(_DWORD *)(v12 + 8 * v19 + 8);
    if ( v20 )
      v20 = a1[17].m128i_i32[1] / v20;
    result = a1[12].m128i_u64[0];
    *(_DWORD *)(result + v19) = v20;
    if ( 4LL * (unsigned int)(v9 - 1) == v19 )
      break;
    v12 = a1[2].m128i_i64[0];
    v19 += 4;
  }
  return result;
}
