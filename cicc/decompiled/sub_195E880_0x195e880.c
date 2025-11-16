// Function: sub_195E880
// Address: 0x195e880
//
__int64 __fastcall sub_195E880(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(200);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_4FB03AC;
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
    *(_QWORD *)v1 = off_49F3BB0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_QWORD *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    *(_DWORD *)(v1 + 184) = 0;
    *(_BYTE *)(v1 + 192) = a1;
    v3 = sub_163A1D0();
    sub_195E660(v3);
  }
  return v2;
}
