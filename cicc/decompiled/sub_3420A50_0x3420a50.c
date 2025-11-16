// Function: sub_3420A50
// Address: 0x3420a50
//
unsigned int __fastcall sub_3420A50(__int64 a1, __int64 a2, _QWORD *a3)
{
  __int128 *v3; // rax
  __int128 *v4; // rax
  __int128 *v5; // rax
  __int128 *v6; // rax

  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
  *(_QWORD *)(a1 + 16) = a2;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 64) = 1;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 120) = 1;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_BYTE *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)a1 = &unk_4A36A08;
  *(_QWORD *)(a1 + 200) = *a3;
  *a3 = 0;
  v3 = sub_BC2B00();
  sub_2DD02F0((__int64)v3);
  v4 = sub_BC2B00();
  sub_FEEC20((__int64)v4);
  v5 = sub_BC2B00();
  sub_CF6DB0((__int64)v5);
  v6 = sub_BC2B00();
  return sub_97FFF0((__int64)v6);
}
