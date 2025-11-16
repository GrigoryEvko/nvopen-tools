// Function: sub_2C46B30
// Address: 0x2c46b30
//
void *__fastcall sub_2C46B30(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v9; // rdi

  v6 = a1 + 8;
  v9 = (__int64)(a1 + 12);
  *(_QWORD *)(v9 - 48) = v6;
  *(_QWORD *)(v9 - 40) = 0x200000000LL;
  *(_QWORD *)(v9 - 72) = 0;
  *(_QWORD *)(v9 - 64) = 0;
  *(_QWORD *)(v9 - 96) = &unk_4A23A70;
  *(_QWORD *)(v9 - 56) = &unk_4A23AA8;
  *(_BYTE *)(v9 - 88) = 2;
  *(_QWORD *)(v9 - 80) = 0;
  *(_QWORD *)(v9 - 16) = 0;
  *(_QWORD *)(v9 - 8) = 0;
  sub_2BF0340(v9, 1, 0, (__int64)a1, a5, a6);
  *a1 = &unk_4A231C8;
  a1[5] = &unk_4A23200;
  a1[12] = &unk_4A23238;
  a1[19] = a2;
  a1[20] = a3;
  *a1 = &unk_4A24AB8;
  a1[5] = &unk_4A24AF0;
  a1[12] = &unk_4A24B28;
  return &unk_4A24B28;
}
