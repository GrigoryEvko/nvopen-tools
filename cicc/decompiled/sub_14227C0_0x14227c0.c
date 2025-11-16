// Function: sub_14227C0
// Address: 0x14227c0
//
__int64 __fastcall sub_14227C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  _QWORD v8[4]; // [rsp+0h] [rbp-20h] BYREF

  v8[1] = a1;
  v6 = *(_QWORD *)(a1 + 16);
  v8[0] = &unk_49EB1F8;
  ((void (__fastcall *)(__int64, __int64, _QWORD *, _QWORD, _QWORD, __int64))sub_1559E80)(v6, a2, v8, 0, 0, a6);
  v8[0] = &unk_49EB1F8;
  return nullsub_544(v8);
}
