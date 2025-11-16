// Function: sub_17CCCF0
// Address: 0x17cccf0
//
__int64 sub_17CCCF0()
{
  __int64 result; // rax
  int v1; // edx
  bool v2; // sf
  int v3; // edx
  char v4; // dl

  result = sub_22077B0(448);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 80) = result + 64;
    *(_QWORD *)(result + 88) = result + 64;
    *(_QWORD *)(result + 128) = result + 112;
    *(_QWORD *)(result + 136) = result + 112;
    v1 = dword_4FA4DE0;
    *(_QWORD *)(result + 16) = &unk_4FA3E5C;
    v2 = v1 < 0;
    v3 = 0;
    if ( !v2 )
      v3 = dword_4FA4DE0;
    *(_DWORD *)(result + 24) = 3;
    *(_QWORD *)(result + 32) = 0;
    *(_DWORD *)(result + 156) = v3;
    v4 = byte_4FA4D00;
    *(_QWORD *)(result + 40) = 0;
    *(_QWORD *)(result + 48) = 0;
    *(_DWORD *)(result + 64) = 0;
    *(_QWORD *)(result + 72) = 0;
    *(_QWORD *)(result + 96) = 0;
    *(_DWORD *)(result + 112) = 0;
    *(_QWORD *)(result + 120) = 0;
    *(_QWORD *)(result + 144) = 0;
    *(_BYTE *)(result + 152) = 0;
    *(_QWORD *)result = off_49F0408;
    *(_BYTE *)(result + 160) = v4;
    *(_BYTE *)(result + 248) = 0;
  }
  return result;
}
