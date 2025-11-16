// Function: sub_1B6D360
// Address: 0x1b6d360
//
__int64 __fastcall sub_1B6D360(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax

  v2 = sub_22077B0(184);
  v4 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4FB76AC;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_QWORD *)(v2 + 168) = v2 + 160;
    *(_QWORD *)(v2 + 160) = v2 + 160;
    *(_QWORD *)v2 = off_49F68B8;
    *(_DWORD *)(v2 + 24) = 5;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)(v2 + 176) = 0;
    sub_1B6D120(v2 + 160, a2, v3);
    v5 = sub_163A1D0();
    sub_1B698C0(v5);
  }
  return v4;
}
