// Function: sub_1483C80
// Address: 0x1483c80
//
__int64 __fastcall sub_1483C80(_QWORD *a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v6; // rax
  __int64 v7; // rbx

  v6 = sub_1456040(a2);
  v7 = sub_1456C90((__int64)a1, v6);
  if ( v7 == sub_1456C90((__int64)a1, a3) )
    return a2;
  else
    return sub_14835F0(a1, a2, a3, 0, a4, a5);
}
