// Function: sub_1484BE0
// Address: 0x1484be0
//
__int64 __fastcall sub_1484BE0(_QWORD *a1, __int64 a2, __int64 a3, char a4, __m128i a5, __m128i a6)
{
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v12; // rax

  v8 = sub_1456040(a3);
  v9 = sub_145CF80((__int64)a1, v8, 1, 0);
  if ( a4 )
  {
    v10 = sub_13A5B00((__int64)a1, a2, a3, 0, 0);
  }
  else
  {
    v12 = sub_14806B0((__int64)a1, a3, v9, 0, 0);
    v10 = sub_13A5B00((__int64)a1, a2, v12, 0, 0);
  }
  return sub_1483CF0(a1, v10, a3, a5, a6);
}
