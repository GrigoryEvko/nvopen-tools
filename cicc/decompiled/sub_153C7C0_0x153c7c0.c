// Function: sub_153C7C0
// Address: 0x153c7c0
//
__int64 __fastcall sub_153C7C0(__int64 a1, __int64 a2, char a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rax

  v6 = sub_22077B0(176);
  v7 = v6;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 8) = 0;
    *(_QWORD *)(v6 + 16) = &unk_4F9DF6C;
    *(_QWORD *)(v6 + 80) = v6 + 64;
    *(_QWORD *)(v6 + 88) = v6 + 64;
    *(_QWORD *)(v6 + 128) = v6 + 112;
    *(_QWORD *)(v6 + 136) = v6 + 112;
    *(_DWORD *)(v6 + 24) = 5;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 0;
    *(_QWORD *)(v6 + 48) = 0;
    *(_DWORD *)(v6 + 64) = 0;
    *(_QWORD *)(v6 + 72) = 0;
    *(_QWORD *)(v6 + 96) = 0;
    *(_DWORD *)(v6 + 112) = 0;
    *(_QWORD *)(v6 + 120) = 0;
    *(_QWORD *)(v6 + 144) = 0;
    *(_BYTE *)(v6 + 152) = 0;
    *(_QWORD *)v6 = off_49ECD58;
    *(_QWORD *)(v6 + 160) = a1;
    *(_BYTE *)(v6 + 168) = a2;
    *(_BYTE *)(v6 + 169) = a3;
    *(_BYTE *)(v6 + 170) = a4;
    v8 = sub_163A1D0(176, a2);
    sub_153C5E0(v8);
  }
  return v7;
}
