// Function: sub_1824BF0
// Address: 0x1824bf0
//
__int64 sub_1824BF0()
{
  __int64 result; // rax
  bool v1; // zf
  int v2; // edx

  result = sub_22077B0(472);
  if ( result )
  {
    v1 = byte_4FA99A0 == 0;
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 80) = result + 64;
    *(_QWORD *)(result + 88) = result + 64;
    *(_QWORD *)(result + 128) = result + 112;
    *(_QWORD *)(result + 136) = result + 112;
    v2 = 1;
    *(_QWORD *)(result + 16) = &unk_4FA93AC;
    *(_DWORD *)(result + 24) = 5;
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
    *(_QWORD *)result = off_49F0A40;
    if ( v1 )
      v2 = 1 - ((byte_4FA98C0 == 0) - 1);
    *(_DWORD *)(result + 156) = v2;
    *(_DWORD *)(result + 400) = 0;
    *(_QWORD *)(result + 408) = 0;
    *(_QWORD *)(result + 416) = result + 400;
    *(_QWORD *)(result + 424) = result + 400;
    *(_QWORD *)(result + 432) = 0;
  }
  return result;
}
