// Function: sub_15E8E70
// Address: 0x15e8e70
//
__int64 __fastcall sub_15E8E70(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r12
  __int64 v5; // rax

  v2 = sub_22077B0(200);
  v4 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4F9E22C;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_DWORD *)(v2 + 24) = 0;
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
    *(_QWORD *)v2 = off_49ED328;
    v5 = sub_16BA580(200, a2, v3);
    *(_QWORD *)(v4 + 176) = 0;
    *(_QWORD *)(v4 + 160) = v5;
    *(_QWORD *)(v4 + 168) = v4 + 184;
    *(_BYTE *)(v4 + 184) = 0;
  }
  return v4;
}
