// Function: sub_1CCF890
// Address: 0x1ccf890
//
__int64 sub_1CCF890()
{
  __int64 result; // rax

  result = sub_22077B0(208);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 80) = result + 64;
    *(_QWORD *)(result + 88) = result + 64;
    *(_QWORD *)(result + 128) = result + 112;
    *(_QWORD *)(result + 136) = result + 112;
    *(_QWORD *)(result + 16) = &unk_4FBF390;
    *(_DWORD *)(result + 24) = 3;
    *(_QWORD *)(result + 32) = 0;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_DWORD *)(result + 64) = 0;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_DWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 120) = 0;
    *(_QWORD *)(result + 144) = 0;
    *(_BYTE *)(result + 152) = 0;
    *(_QWORD *)result = off_49F8C68;
    *(_QWORD *)(result + 160) = result + 176;
    *(_QWORD *)(result + 168) = 0x400000000LL;
  }
  return result;
}
