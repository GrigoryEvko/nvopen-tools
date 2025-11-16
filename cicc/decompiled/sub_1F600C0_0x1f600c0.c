// Function: sub_1F600C0
// Address: 0x1f600c0
//
__int64 sub_1F600C0()
{
  __int64 result; // rax

  result = sub_22077B0(256);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 80) = result + 64;
    *(_QWORD *)(result + 88) = result + 64;
    *(_QWORD *)(result + 128) = result + 112;
    *(_QWORD *)(result + 136) = result + 112;
    *(_QWORD *)(result + 16) = &unk_4FCE434;
    *(_DWORD *)(result + 24) = 3;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_DWORD *)(result + 64) = 0;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_DWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 120) = 0;
    *(_QWORD *)result = off_49FFDC0;
    *(_QWORD *)(result + 144) = 0;
    *(_WORD *)(result + 152) = 0;
    *(_QWORD *)(result + 156) = 0;
    *(_QWORD *)(result + 164) = 0;
    *(_QWORD *)(result + 172) = 0;
    *(_QWORD *)(result + 180) = 0;
    *(_QWORD *)(result + 188) = 0;
    *(_QWORD *)(result + 200) = 0;
    *(_QWORD *)(result + 208) = 0;
    *(_QWORD *)(result + 216) = 0;
    *(_DWORD *)(result + 224) = 0;
    *(_QWORD *)(result + 232) = 0;
    *(_QWORD *)(result + 240) = 0;
    *(_QWORD *)(result + 248) = 0;
  }
  return result;
}
