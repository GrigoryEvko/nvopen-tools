// Function: sub_14241E0
// Address: 0x14241e0
//
__int64 __fastcall sub_14241E0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __m128i *v3; // r14
  unsigned __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx
  __m128i *v7; // rbx
  __int64 i; // rdx
  __m128i *v9; // r15
  __m128i *v10; // rax
  __int8 v11; // dl
  __int64 j; // rdx
  __m128i *v13; // [rsp+18h] [rbp-98h] BYREF
  _QWORD v14[6]; // [rsp+20h] [rbp-90h] BYREF
  _QWORD v15[12]; // [rsp+50h] [rbp-60h] BYREF

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(__m128i **)(a1 + 8);
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
      | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
      | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
      | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
      | (unsigned int)(a2 - 1)
      | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
     + 1;
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = sub_22077B0(96LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v3[6 * v2];
    for ( i = result + 96 * v6; i != result; result += 96 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = -8;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 40) = 0;
      }
    }
    v14[0] = 0;
    v14[1] = -8;
    memset(&v14[2], 0, 32);
    v15[0] = 0;
    v15[1] = -16;
    memset(&v15[2], 0, 32);
    if ( v7 != v3 )
    {
      v9 = v3;
      do
      {
        while ( (unsigned __int8)sub_1423BB0(v9, (__int64)v14) || (unsigned __int8)sub_1423BB0(v9, (__int64)v15) )
        {
          v9 += 6;
          if ( v7 == v9 )
            return j___libc_free_0(v3);
        }
        sub_1423F10(a1, (__int64)v9, &v13);
        v10 = v13;
        *v13 = _mm_loadu_si128(v9);
        v10[1] = _mm_loadu_si128(v9 + 1);
        v10[2] = _mm_loadu_si128(v9 + 2);
        v10[3].m128i_i64[0] = v9[3].m128i_i64[0];
        v10[3].m128i_i64[1] = v9[3].m128i_i64[1];
        v10[4].m128i_i64[0] = v9[4].m128i_i64[0];
        v10[4].m128i_i64[1] = v9[4].m128i_i64[1];
        v10[5].m128i_i64[0] = v9[5].m128i_i64[0];
        v10[5].m128i_i8[8] = v9[5].m128i_i8[8];
        v11 = v9[5].m128i_i8[10];
        v10[5].m128i_i8[10] = v11;
        if ( v11 )
          v10[5].m128i_i8[9] = v9[5].m128i_i8[9];
        v9 += 6;
        ++*(_DWORD *)(a1 + 16);
      }
      while ( v7 != v9 );
    }
    return j___libc_free_0(v3);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = result + 96LL * *(unsigned int *)(a1 + 24); j != result; result += 96 )
    {
      if ( result )
      {
        *(_BYTE *)result = 0;
        *(_QWORD *)(result + 8) = -8;
        *(_QWORD *)(result + 16) = 0;
        *(_QWORD *)(result + 24) = 0;
        *(_QWORD *)(result + 32) = 0;
        *(_QWORD *)(result + 40) = 0;
      }
    }
  }
  return result;
}
