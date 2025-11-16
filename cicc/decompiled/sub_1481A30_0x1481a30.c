// Function: sub_1481A30
// Address: 0x1481a30
//
__int64 __fastcall sub_1481A30(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v12; // rax

  v6 = a2;
  v7 = sub_1456040(a2);
  v8 = sub_1456C90((__int64)a1, v7);
  v9 = sub_1456040(a3);
  if ( v8 <= sub_1456C90((__int64)a1, v9) )
  {
    v12 = sub_1456040(a3);
    v6 = sub_14758B0((__int64)a1, a2, v12);
  }
  else
  {
    v10 = sub_1456040(a2);
    a3 = sub_14747F0((__int64)a1, a3, v10, 0);
  }
  return sub_14819D0(a1, v6, a3, a4, a5);
}
