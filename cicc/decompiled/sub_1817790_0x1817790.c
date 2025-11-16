// Function: sub_1817790
// Address: 0x1817790
//
__int64 __fastcall sub_1817790(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  unsigned __int64 v4; // r12
  __int64 result; // rax
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rsi
  char v9; // cl

  sub_16CCCB0(a1, (__int64)(a1 + 5), a2);
  v4 = *(_QWORD *)(a2 + 112) - *(_QWORD *)(a2 + 104);
  a1[13] = 0;
  a1[14] = 0;
  a1[15] = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFE0LL )
      sub_4261EA(a1, a1 + 5, v3);
    result = sub_22077B0(v4);
  }
  else
  {
    v4 = 0;
    result = 0;
  }
  a1[13] = result;
  a1[14] = result;
  a1[15] = result + v4;
  v6 = *(_QWORD *)(a2 + 112);
  v7 = *(_QWORD *)(a2 + 104);
  if ( v6 == v7 )
  {
    a1[14] = result;
  }
  else
  {
    v8 = result + v6 - v7;
    do
    {
      if ( result )
      {
        *(_QWORD *)result = *(_QWORD *)v7;
        v9 = *(_BYTE *)(v7 + 24);
        *(_BYTE *)(result + 24) = v9;
        if ( v9 )
          *(__m128i *)(result + 8) = _mm_loadu_si128((const __m128i *)(v7 + 8));
      }
      result += 32;
      v7 += 32;
    }
    while ( result != v8 );
    a1[14] = v8;
  }
  return result;
}
