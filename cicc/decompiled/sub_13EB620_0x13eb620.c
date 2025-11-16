// Function: sub_13EB620
// Address: 0x13eb620
//
void __fastcall sub_13EB620(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax

  if ( a1[4] )
  {
    v6 = sub_157EB90(a2);
    v7 = sub_1632FA0(v6);
    v8 = sub_13E7A30(a1 + 4, *a1, v7, a1[3]);
    sub_13E7B40(v8, a3, a4);
  }
}
