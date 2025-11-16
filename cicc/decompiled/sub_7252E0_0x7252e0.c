// Function: sub_7252E0
// Address: 0x7252e0
//
__int64 __fastcall sub_7252E0(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 168);
  *(_WORD *)(result + 44) = -1;
  *(_BYTE *)(result + 92) &= 0xE2u;
  *(_QWORD *)(result + 104) &= 0x1FE80000000000uLL;
  *(_WORD *)(result + 112) &= 0xF8u;
  *(_QWORD *)result = 0;
  *(_QWORD *)(result + 8) = 0;
  *(_QWORD *)(result + 16) = 0;
  *(_QWORD *)(result + 24) = 0;
  *(_QWORD *)(result + 32) = 0;
  *(_DWORD *)(result + 40) = 1;
  *(_QWORD *)(result + 48) = -3;
  *(_QWORD *)(result + 56) = 0;
  *(_QWORD *)(result + 64) = 0;
  *(_QWORD *)(result + 72) = 0;
  *(_QWORD *)(result + 80) = 0;
  *(_DWORD *)(result + 88) = 0;
  *(_QWORD *)(result + 96) = 0;
  *(_QWORD *)(result + 120) = 0;
  *(_QWORD *)(result + 136) = 0;
  *(_QWORD *)(result + 144) = 0;
  *(_QWORD *)(result + 152) = 0;
  *(_QWORD *)(result + 176) = 0;
  *(_QWORD *)(result + 184) = 0;
  *(_QWORD *)(result + 192) = 0;
  *(_QWORD *)(result + 200) = 0;
  *(_QWORD *)(result + 208) = 0;
  *(_QWORD *)(result + 216) = 0;
  *(_QWORD *)(result + 224) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_BYTE *)(a1 + 141) |= 0x20u;
  *(_DWORD *)(a1 + 176) &= 0xFCFFFC0D;
  *(_DWORD *)(a1 + 136) = 1;
  *(_QWORD *)(a1 + 160) = 0;
  return result;
}
