// Function: sub_1F45E70
// Address: 0x1f45e70
//
void __fastcall __noreturn sub_1F45E70(__int64 a1)
{
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = &unk_4FCBA30;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)a1 = &unk_49FF4C8;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_WORD *)(a1 + 200) = 1;
  *(_BYTE *)(a1 + 202) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 0x10000;
  sub_16BD130(
    "Trying to construct TargetPassConfig without a target machine. Scheduling a CodeGen pass without a target triple set?",
    1u);
}
