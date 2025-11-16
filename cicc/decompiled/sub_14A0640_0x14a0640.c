// Function: sub_14A0640
// Address: 0x14a0640
//
__int64 __fastcall sub_14A0640(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  *(_QWORD *)(a1 + 16) = &unk_4F9B6E8;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 5;
  *(_QWORD *)a1 = &unk_49EC9B0;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  sub_14A0550(a1 + 160);
  *(_QWORD *)(a1 + 360) = a1 + 160;
  v2 = sub_163A1D0(a1 + 160, a2);
  return sub_149CBF0(v2);
}
