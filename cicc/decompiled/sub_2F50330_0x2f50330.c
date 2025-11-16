// Function: sub_2F50330
// Address: 0x2f50330
//
unsigned int __fastcall sub_2F50330(__int64 a1, __int64 a2)
{
  void (__fastcall *v2)(__int64, __int64, __int64); // rax
  __int128 *v3; // rax

  *(_QWORD *)(a1 + 16) = &unk_5023990;
  *(_QWORD *)(a1 + 56) = a1 + 104;
  *(_QWORD *)(a1 + 112) = a1 + 160;
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
  *(_QWORD *)a1 = off_4A2B0E0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 88) = 1065353216;
  *(_DWORD *)(a1 + 144) = 1065353216;
  v2 = *(void (__fastcall **)(__int64, __int64, __int64))(a2 + 16);
  if ( v2 )
  {
    v2(a1 + 200, a2, 2);
    *(_QWORD *)(a1 + 224) = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a1 + 216) = *(_QWORD *)(a2 + 16);
  }
  v3 = sub_BC2B00();
  return sub_2F502B0((__int64)v3);
}
