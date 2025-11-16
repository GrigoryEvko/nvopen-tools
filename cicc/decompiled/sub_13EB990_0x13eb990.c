// Function: sub_13EB990
// Address: 0x13eb990
//
void __fastcall sub_13EB990(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  _QWORD v7[11]; // [rsp-58h] [rbp-58h] BYREF

  if ( a1[4] )
  {
    v6 = sub_13E7A30(a1 + 4, *a1, a1[1], a1[3]);
    v7[0] = off_49EA9C0;
    v7[1] = v6;
    v7[2] = a3;
    ((void (__fastcall *)(__int64, __int64, _QWORD *, _QWORD, _QWORD))sub_1559E80)(a2, a4, v7, 0, 0);
    v7[0] = off_49EA9C0;
    nullsub_544(v7);
  }
}
