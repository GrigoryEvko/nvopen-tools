// Function: sub_8F0790
// Address: 0x8f0790
//
__int64 __fastcall sub_8F0790(__int64 a1, unsigned __int64 a2)
{
  int v3; // eax
  int v4; // ecx
  int v5; // ecx
  unsigned __int64 v6; // rdx
  __int64 v7; // rax
  int v8; // r8d
  bool v9; // di
  int v10; // edx
  int v11; // r9d
  unsigned __int64 v12; // rbx
  __int64 result; // rax
  __m128i v14; // xmm2
  __m128i v15; // [rsp+0h] [rbp-40h] BYREF
  __m128i v16; // [rsp+10h] [rbp-30h] BYREF

  v3 = *(_DWORD *)(a1 + 28);
  v15 = 0;
  v16.m128i_i64[0] = 0;
  v4 = v3 + 14;
  v16.m128i_i32[3] = v3;
  v16.m128i_i32[2] = 0;
  if ( v3 + 7 >= 0 )
    v4 = v3 + 7;
  v15.m128i_i32[0] = 2;
  v5 = v4 >> 3;
  if ( v3 > 0 && (v6 = a2, v7 = 1, a2) )
  {
    do
    {
      v15.m128i_i8[v7 + 11] = v6;
      v8 = v7;
      v9 = v5 <= (int)v7;
      v6 >>= 8;
      ++v7;
    }
    while ( v6 != 0 && !v9 );
  }
  else
  {
    v8 = 0;
  }
  if ( v5 > v8 )
    memset((char *)&v15.m128i_u64[1] + v8 + 4, 0, (unsigned int)(v5 - 1 - v8) + 1LL);
  v10 = sub_8EE4D0(&v15.m128i_i8[12], v16.m128i_i32[3]);
  if ( !v10 && (v12 = (unsigned __int64)((-1 << v11) & (unsigned int)a2) >> v11) != 0 )
  {
    do
    {
      ++v10;
      v12 >>= 1;
    }
    while ( v12 );
    result = (unsigned int)(v11 + v10 - 1);
    v15.m128i_i32[2] = v11 + v10 - 1;
  }
  else
  {
    v15.m128i_i32[2] = v11 - v10;
    result = sub_8EE880(&v15.m128i_i8[12], v11, v10);
  }
  if ( v15.m128i_i32[0] != 6 )
  {
    if ( *(_DWORD *)a1 == 6 )
    {
      v14 = _mm_loadu_si128(&v16);
      *(__m128i *)a1 = _mm_loadu_si128(&v15);
      *(__m128i *)(a1 + 16) = v14;
    }
    else if ( *(_DWORD *)(a1 + 8) >= v15.m128i_i32[2] )
    {
      return (__int64)sub_8EF770((_DWORD *)a1, (_DWORD *)a1, (__int64)&v15);
    }
    else
    {
      return (__int64)sub_8EF770((_DWORD *)a1, &v15, a1);
    }
  }
  return result;
}
