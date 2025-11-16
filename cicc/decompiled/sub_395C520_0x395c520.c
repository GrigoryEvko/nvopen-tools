// Function: sub_395C520
// Address: 0x395c520
//
__int64 __fastcall sub_395C520(int a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(0xA0u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_5054870;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_DWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_QWORD *)v1 = off_4A3F340;
    *(_DWORD *)(v1 + 156) = a1;
    v3 = sub_163A1D0();
    sub_395C350(v3);
  }
  return v2;
}
