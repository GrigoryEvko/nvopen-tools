// Function: sub_2FAEDD0
// Address: 0x2faedd0
//
__int64 sub_2FAEDD0()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(0x1E8u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 16) = &unk_5025C34;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)(v0 + 32) = 0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    *(_BYTE *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)v0 = &unk_4A2C0C0;
    *(_DWORD *)(v0 + 144) = 1065353216;
    sub_2FAED30(v0 + 200);
  }
  return v1;
}
