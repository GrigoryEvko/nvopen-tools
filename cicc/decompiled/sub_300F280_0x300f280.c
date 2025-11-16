// Function: sub_300F280
// Address: 0x300f280
//
__int64 sub_300F280()
{
  __int64 result; // rax

  result = sub_22077B0(0x118u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 56) = result + 104;
    *(_QWORD *)(result + 16) = &unk_502A674;
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
    *(_QWORD *)result = off_4A2E010;
    *(_QWORD *)(result + 176) = 0;
    *(_QWORD *)(result + 184) = 0;
    *(_QWORD *)(result + 192) = 0;
    *(_QWORD *)(result + 200) = 0;
    *(_QWORD *)(result + 208) = 0;
    *(_QWORD *)(result + 216) = 0;
    *(_QWORD *)(result + 224) = 0;
    *(_QWORD *)(result + 232) = 0;
    *(_QWORD *)(result + 240) = 0;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
    *(_QWORD *)(result + 248) = 0;
    *(_QWORD *)(result + 256) = 0;
    *(_QWORD *)(result + 264) = 0;
    *(_QWORD *)(result + 272) = 0;
  }
  return result;
}
