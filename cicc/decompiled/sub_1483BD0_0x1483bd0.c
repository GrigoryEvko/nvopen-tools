// Function: sub_1483BD0
// Address: 0x1483bd0
//
__int64 __fastcall sub_1483BD0(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // r15
  __int64 v7; // rbx
  unsigned __int64 v8; // rbx

  v6 = sub_1456040(a2);
  v7 = sub_1456C90((__int64)a1, v6);
  if ( v7 == sub_1456C90((__int64)a1, a3) )
    return a2;
  v8 = sub_1456C90((__int64)a1, v6);
  if ( v8 <= sub_1456C90((__int64)a1, a3) )
    return sub_147B0D0((__int64)a1, a2, a3, 0);
  else
    return sub_14835F0(a1, a2, a3, 0, a4, a5);
}
