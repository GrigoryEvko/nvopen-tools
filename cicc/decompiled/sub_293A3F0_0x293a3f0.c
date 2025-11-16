// Function: sub_293A3F0
// Address: 0x293a3f0
//
__int64 sub_293A3F0()
{
  __int64 result; // rax

  result = sub_22077B0(0xD8u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 56) = result + 104;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
    *(_OWORD *)(result + 176) = 0;
    *(_QWORD *)(result + 16) = &unk_5005714;
    *(_DWORD *)(result + 24) = 2;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_QWORD *)(result + 64) = 1;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 80) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_QWORD *)(result + 104) = 0;
    *(_QWORD *)(result + 112) = result + 160;
    *(_QWORD *)(result + 120) = 1;
    *(_QWORD *)(result + 128) = 0;
    *(_QWORD *)(result + 136) = 0;
    *(_QWORD *)(result + 152) = 0;
    *(_QWORD *)(result + 160) = 0;
    *(_BYTE *)(result + 168) = 0;
    *(_QWORD *)result = off_4A21F48;
    *(_QWORD *)(result + 208) = 0;
    *(_BYTE *)(result + 180) = 1;
    *(_OWORD *)(result + 192) = 0;
  }
  return result;
}
