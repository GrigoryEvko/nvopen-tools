// Function: sub_1705F50
// Address: 0x1705f50
//
__int64 sub_1705F50()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rax

  v0 = sub_22077B0(2264);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 16) = &unk_4FA1824;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)v0 = &unk_49EFFD8;
    *(_QWORD *)(v0 + 160) = v0 + 176;
    *(_QWORD *)(v0 + 168) = 0x10000000000LL;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 2224) = 0;
    *(_QWORD *)(v0 + 2232) = 0;
    *(_QWORD *)(v0 + 2240) = 0;
    *(_DWORD *)(v0 + 2248) = 0;
    *(_WORD *)(v0 + 2256) = 1;
    v2 = sub_163A1D0();
    sub_1705E40(v2);
  }
  return v1;
}
