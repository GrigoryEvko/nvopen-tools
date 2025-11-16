// Function: sub_16809A0
// Address: 0x16809a0
//
__int64 __fastcall sub_16809A0(_QWORD *a1, __int64 a2, _QWORD *a3, __int64 a4)
{
  _QWORD *v4; // r13
  __int64 v6; // r8
  _QWORD *v7; // rbx
  __int64 v8; // rdx
  __int64 result; // rax
  __int64 v10; // [rsp+8h] [rbp-58h]
  __m128i v11; // [rsp+10h] [rbp-50h]

  v4 = &a3[8 * a4];
  if ( a3 != v4 )
  {
    v6 = a4;
    v7 = a3;
    do
    {
      v8 = v7[2];
      if ( *(_QWORD *)(a2 + 16) != v8
        || *(_QWORD *)(a2 + 24) != v7[3]
        || (result = v7[4], *(_QWORD *)(a2 + 32) != result) )
      {
        v11 = _mm_loadu_si128((const __m128i *)(a2 + 40));
        result = v7[4] & *(_QWORD *)(a2 + 56) | v8 & v11.m128i_i64[0] | v7[3] & v11.m128i_i64[1];
        if ( result )
        {
          *a1 |= v8;
          a1[1] |= v7[3];
          a1[2] |= v7[4];
          v10 = v6;
          result = sub_16809A0(a1, v7, a3, v6);
          v6 = v10;
        }
      }
      v7 += 8;
    }
    while ( v4 != v7 );
  }
  return result;
}
