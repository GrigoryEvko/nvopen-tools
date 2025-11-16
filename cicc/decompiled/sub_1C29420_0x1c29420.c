// Function: sub_1C29420
// Address: 0x1c29420
//
__int64 __fastcall sub_1C29420(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  _BYTE *v5; // rsi
  __int64 v6; // rdx

  v3 = sub_22077B0(200);
  v4 = v3;
  if ( v3 )
  {
    v5 = *(_BYTE **)a2;
    *(_QWORD *)(v3 + 8) = 0;
    v6 = *(_QWORD *)(a2 + 8);
    *(_DWORD *)(v3 + 24) = 2;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4FBA374;
    *(_QWORD *)(v3 + 80) = v3 + 64;
    *(_QWORD *)(v3 + 88) = v3 + 64;
    *(_QWORD *)(v3 + 128) = v3 + 112;
    *(_QWORD *)(v3 + 136) = v3 + 112;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)v3 = &unk_49F7700;
    *(_DWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_DWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_BYTE *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = a1;
    *(_QWORD *)(v3 + 168) = v3 + 184;
    sub_1C286E0((__int64 *)(v3 + 168), v5, (__int64)&v5[v6]);
  }
  return v4;
}
