// Function: sub_29C1AA0
// Address: 0x29c1aa0
//
__int64 sub_29C1AA0()
{
  __int64 result; // rax

  result = sub_22077B0(0xE8u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 56) = result + 104;
    *(_QWORD *)(result + 112) = result + 160;
    *(_QWORD *)(result + 16) = &unk_5008BB8;
    *(_DWORD *)(result + 24) = 4;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_QWORD *)(result + 64) = 1;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 80) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 120) = 1;
    *(_QWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_QWORD *)(result + 160) = 0;
    *(_BYTE *)(result + 168) = 0;
    *(_QWORD *)result = off_4A22838;
    *(_QWORD *)(result + 176) = byte_3F871B3;
    *(_QWORD *)(result + 184) = 0;
    *(_QWORD *)(result + 192) = byte_3F871B3;
    *(_QWORD *)(result + 200) = 0;
    *(_QWORD *)(result + 208) = 0;
    *(_QWORD *)(result + 216) = 0;
    *(_DWORD *)(result + 224) = 1;
    *(_BYTE *)(result + 228) = 0;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
  }
  return result;
}
