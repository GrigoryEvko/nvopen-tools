// Function: sub_198DF00
// Address: 0x198df00
//
__int64 __fastcall sub_198DF00(int a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(160);
  v2 = v1;
  if ( !v1 )
    return v2;
  *(_QWORD *)(v1 + 8) = 0;
  *(_QWORD *)(v1 + 16) = &unk_4FB0E2C;
  *(_QWORD *)(v1 + 80) = v1 + 64;
  *(_QWORD *)(v1 + 88) = v1 + 64;
  *(_QWORD *)(v1 + 128) = v1 + 112;
  *(_QWORD *)(v1 + 136) = v1 + 112;
  *(_DWORD *)(v1 + 24) = 2;
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
  *(_QWORD *)v1 = off_49F41E0;
  v3 = sub_163A1D0();
  sub_198DD20(v3);
  if ( a1 != -1 )
  {
    *(_DWORD *)(v2 + 156) = a1;
    return v2;
  }
  *(_DWORD *)(v2 + 156) = dword_4FB0EE0;
  return v2;
}
