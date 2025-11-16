// Function: sub_23EA340
// Address: 0x23ea340
//
__int64 __fastcall sub_23EA340(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int64 v9; // rax
  _QWORD v11[8]; // [rsp+10h] [rbp-40h] BYREF

  v11[0] = a6;
  v11[1] = a7;
  v9 = sub_BCF480(a5, v11, 2, 0);
  return sub_BA8C10(a1, a2, a3, v9, a4);
}
