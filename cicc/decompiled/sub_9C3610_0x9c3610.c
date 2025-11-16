// Function: sub_9C3610
// Address: 0x9c3610
//
__int64 __fastcall sub_9C3610(__int64 a1, __int64 a2)
{
  _QWORD v3[12]; // [rsp+0h] [rbp-60h] BYREF

  *(_QWORD *)a1 = a1 + 16;
  v3[5] = 0x100000000LL;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  v3[6] = a1;
  v3[0] = &unk_49DD210;
  memset(&v3[1], 0, 32);
  sub_CB5980(v3, 0, 0, 0);
  (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a2 + 16LL))(a2, v3);
  v3[0] = &unk_49DD210;
  sub_CB5840(v3);
  return a1;
}
