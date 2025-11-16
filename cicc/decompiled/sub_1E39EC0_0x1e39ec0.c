// Function: sub_1E39EC0
// Address: 0x1e39ec0
//
__int64 __fastcall sub_1E39EC0(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int64 v3; // rax

  v1 = sub_22077B0(184);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_4FC6AF4;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_QWORD *)v1 = off_49FC080;
    *(_DWORD *)(v1 + 24) = 5;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_WORD *)(v1 + 152) = 0;
    *(_BYTE *)(v1 + 154) = 1;
    *(_QWORD *)(v1 + 160) = 0;
    *(_QWORD *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    v3 = sub_163A1D0();
    sub_1E39DE0(v3);
  }
  *(_BYTE *)(v2 + 154) = a1;
  return v2;
}
