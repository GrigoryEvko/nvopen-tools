// Function: sub_27EE150
// Address: 0x27ee150
//
__int64 __fastcall sub_27EE150(char a1)
{
  int v1; // r13d
  int v2; // r14d
  __int64 v3; // rax
  __int64 v4; // r12
  __int128 *v5; // rax

  v1 = qword_4FFDE88[8];
  v2 = qword_4FFDDA8[8];
  v3 = sub_22077B0(0xB8u);
  v4 = v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4FFDD4C;
    *(_QWORD *)(v3 + 56) = v3 + 104;
    *(_QWORD *)(v3 + 112) = v3 + 160;
    *(_DWORD *)(v3 + 24) = 1;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 64) = 1;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 80) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 104) = 0;
    *(_QWORD *)(v3 + 120) = 1;
    *(_QWORD *)(v3 + 128) = 0;
    *(_QWORD *)(v3 + 136) = 0;
    *(_QWORD *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = 0;
    *(_BYTE *)(v3 + 168) = 0;
    *(_QWORD *)v3 = off_4A21048;
    *(_DWORD *)(v3 + 172) = v1;
    *(_DWORD *)(v3 + 176) = v2;
    *(_BYTE *)(v3 + 180) = 1;
    *(_BYTE *)(v3 + 181) = a1;
    *(_DWORD *)(v3 + 88) = 1065353216;
    *(_DWORD *)(v3 + 144) = 1065353216;
    v5 = sub_BC2B00();
    sub_27EDF70((__int64)v5);
  }
  return v4;
}
