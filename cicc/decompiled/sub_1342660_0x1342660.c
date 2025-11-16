// Function: sub_1342660
// Address: 0x1342660
//
char __fastcall sub_1342660(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r15
  __int64 v9; // [rsp+8h] [rbp-1B8h]
  _QWORD v10[54]; // [rsp+10h] [rbp-1B0h] BYREF

  v5 = (_QWORD *)(a1 + 432);
  if ( !a1 )
  {
    v5 = v10;
    v9 = a4;
    sub_130D500(v10);
    a4 = v9;
  }
  sub_1341260(a1, a2, v5, a4, 1, 0, (__int64 *)a3, (unsigned __int64 *)(a3 + 8));
  return sub_1341260(a1, a2, v5, a5, 1, 0, (__int64 *)(a3 + 16), (unsigned __int64 *)(a3 + 24));
}
