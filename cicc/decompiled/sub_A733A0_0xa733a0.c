// Function: sub_A733A0
// Address: 0xa733a0
//
__int64 __fastcall sub_A733A0(__int64 a1, int a2)
{
  int v2; // ecx
  __int64 *v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 *v8; // r14
  __int64 v9; // rax
  __m128i v10; // [rsp+20h] [rbp-50h] BYREF
  __int128 v11; // [rsp+30h] [rbp-40h] BYREF

  v2 = *(unsigned __int8 *)(a1 + a2 / 8 + 12);
  if ( _bittest(&v2, a2 % 8) )
  {
    v4 = (__int64 *)(a1 + 64);
    v5 = 8LL * *(unsigned int *)(a1 + 48);
    v6 = (8LL * *(unsigned int *)(a1 + 8) - v5) >> 3;
    if ( 8LL * *(unsigned int *)(a1 + 8) + 64 - v5 > 64 )
    {
      do
      {
        while ( 1 )
        {
          v7 = v6 >> 1;
          v8 = &v4[v6 >> 1];
          *(_QWORD *)&v11 = *v8;
          if ( a2 <= (int)sub_A71AE0((__int64 *)&v11) )
            break;
          v4 = v8 + 1;
          v6 = v6 - v7 - 1;
          if ( v6 <= 0 )
            goto LABEL_9;
        }
        v6 >>= 1;
      }
      while ( v7 > 0 );
    }
LABEL_9:
    v9 = *v4;
    v10.m128i_i8[8] = 1;
    v10.m128i_i64[0] = v9;
    return _mm_loadu_si128(&v10).m128i_i64[0];
  }
  else
  {
    BYTE8(v11) = 0;
  }
  return v11;
}
