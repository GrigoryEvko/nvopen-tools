// Function: sub_36CD270
// Address: 0x36cd270
//
__int64 __fastcall sub_36CD270(char a1, char a2)
{
  __int64 result; // rax

  result = sub_22077B0(0xB0u);
  if ( result )
  {
    *(_QWORD *)(result + 8) = 0;
    *(_QWORD *)(result + 16) = &unk_5040830;
    *(_QWORD *)(result + 56) = result + 104;
    *(_DWORD *)(result + 24) = 4;
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
    *(_QWORD *)result = off_4A3AF18;
    *(_BYTE *)(result + 169) = a1;
    *(_BYTE *)(result + 170) = a2;
    *(_DWORD *)(result + 88) = 1065353216;
    *(_DWORD *)(result + 144) = 1065353216;
  }
  return result;
}
