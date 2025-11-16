// Function: sub_214A510
// Address: 0x214a510
//
__int64 __fastcall sub_214A510(__int64 a1)
{
  sub_38DCA60();
  *(_BYTE *)(a1 + 160) = 0;
  *(_QWORD *)a1 = &unk_4A01240;
  *(_QWORD *)(a1 + 16) = a1 + 32;
  *(_QWORD *)(a1 + 24) = 0x400000000LL;
  return 0x400000000LL;
}
