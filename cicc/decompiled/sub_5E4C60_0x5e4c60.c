// Function: sub_5E4C60
// Address: 0x5e4c60
//
__int64 __fastcall sub_5E4C60(__int64 a1, _QWORD *a2)
{
  unsigned __int64 v2; // rax
  __int64 result; // rax

  memset((void *)a1, 0, 0x1D8u);
  *(_QWORD *)(a1 + 152) = a1;
  *(_QWORD *)(a1 + 24) = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    *(_BYTE *)(a1 + 178) |= 1u;
  v2 = (unsigned __int64)(unk_4D04430 & 1) << 27;
  BYTE1(v2) = 64;
  *(_QWORD *)(a1 + 120) = *(_QWORD *)(a1 + 120) & 0xFFFFFFBFF7FFBFFFLL
                        | ((unsigned __int64)(unk_4F0775C & 1) << 38)
                        | v2;
  *(_QWORD *)(a1 + 24) = *a2;
  memset((void *)(a1 + 472), 0, 0x58u);
  *(_QWORD *)(a1 + 576) = 0;
  *(_WORD *)(a1 + 560) = *(_WORD *)(a1 + 560) & 0xC000 | 1;
  result = unk_4F077C8;
  *(_QWORD *)(a1 + 564) = unk_4F077C8;
  return result;
}
