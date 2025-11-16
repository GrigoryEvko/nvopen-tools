// Function: sub_1680CA0
// Address: 0x1680ca0
//
void __fastcall sub_1680CA0(_QWORD *a1, _QWORD *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // [rsp+8h] [rbp-58h]
  __m128i v13; // [rsp+10h] [rbp-50h]

  v4 = a3 + (a4 << 6);
  if ( a3 != v4 )
  {
    v7 = a3;
    do
    {
      v8 = a2[2];
      v9 = *(_QWORD *)(v7 + 16);
      if ( v8 != v9 || a2[3] != *(_QWORD *)(v7 + 24) || a2[4] != *(_QWORD *)(v7 + 32) )
      {
        v13 = _mm_loadu_si128((const __m128i *)(v7 + 40));
        if ( a2[4] & *(_QWORD *)(v7 + 56) | a2[3] & v13.m128i_i64[1] | v13.m128i_i64[0] & v8 )
        {
          v10 = *(_QWORD *)(v7 + 24);
          v11 = *(_QWORD *)(v7 + 32);
          *a1 &= ~v9;
          a1[1] &= ~v10;
          a1[2] &= ~v11;
          v12 = a4;
          sub_1680CA0(a1, v7, a3);
          a4 = v12;
        }
      }
      v7 += 64;
    }
    while ( v4 != v7 );
  }
}
