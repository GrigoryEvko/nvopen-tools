// Function: sub_27EDFF0
// Address: 0x27edff0
//
__int64 sub_27EDFF0()
{
  int v0; // r13d
  int v1; // ebx
  __int64 v2; // rax
  __int64 v3; // r12
  __int128 *v4; // rax

  v0 = qword_4FFDE88[8];
  v1 = qword_4FFDDA8[8];
  v2 = sub_22077B0(0xB8u);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4FFDD4C;
    *(_QWORD *)(v2 + 56) = v2 + 104;
    *(_QWORD *)(v2 + 112) = v2 + 160;
    *(_QWORD *)v2 = off_4A21048;
    *(_DWORD *)(v2 + 24) = 1;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_QWORD *)(v2 + 64) = 1;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 80) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_QWORD *)(v2 + 104) = 0;
    *(_QWORD *)(v2 + 120) = 1;
    *(_QWORD *)(v2 + 128) = 0;
    *(_QWORD *)(v2 + 136) = 0;
    *(_QWORD *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 160) = 0;
    *(_BYTE *)(v2 + 168) = 0;
    *(_DWORD *)(v2 + 172) = v0;
    *(_DWORD *)(v2 + 176) = v1;
    *(_WORD *)(v2 + 180) = 1;
    *(_DWORD *)(v2 + 88) = 1065353216;
    *(_DWORD *)(v2 + 144) = 1065353216;
    v4 = sub_BC2B00();
    sub_27EDF70((__int64)v4);
  }
  return v3;
}
