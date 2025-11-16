// Function: sub_157FB60
// Address: 0x157fb60
//
__int64 __fastcall sub_157FB60(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rax

  v8 = sub_1643280(a2);
  sub_1648CB0(a1, v8, 18);
  a1[3] = 0;
  a1[6] = a1 + 5;
  a1[4] = 0;
  a1[5] = (unsigned __int64)(a1 + 5) | 4;
  a1[7] = 0;
  if ( a4 )
    sub_157FA80((__int64)a1, a4, a5);
  return sub_164B780(a1, a3);
}
