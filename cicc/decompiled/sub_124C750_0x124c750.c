// Function: sub_124C750
// Address: 0x124c750
//
__int64 __fastcall sub_124C750(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, char a5)
{
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_WORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)a1 = &unk_49E66A8;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = *a2;
  *a2 = 0;
  *(_QWORD *)(a1 + 120) = a3;
  *(_QWORD *)(a1 + 128) = a4;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_DWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_DWORD *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 200) = a5;
  *(_BYTE *)(a1 + 201) = 0;
  *(_BYTE *)(a1 + 203) = 0;
  *(_QWORD *)(a1 + 208) = a1 + 224;
  *(_QWORD *)(a1 + 216) = 0;
  return a1 + 224;
}
