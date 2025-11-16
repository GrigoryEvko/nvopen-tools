// Function: sub_B3AAD0
// Address: 0xb3aad0
//
__int64 __fastcall sub_B3AAD0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  _BYTE *v5; // rsi
  __int64 v6; // rdx

  v3 = sub_22077B0(216);
  v4 = v3;
  if ( v3 )
  {
    v5 = *(_BYTE **)a2;
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4F816EC;
    v6 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(v3 + 56) = v3 + 104;
    *(_QWORD *)(v3 + 112) = v3 + 160;
    *(_QWORD *)v3 = off_49DA308;
    *(_DWORD *)(v3 + 24) = 2;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_QWORD *)(v3 + 64) = 1;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 80) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_QWORD *)(v3 + 104) = 0;
    *(_QWORD *)(v3 + 120) = 1;
    *(_QWORD *)(v3 + 128) = 0;
    *(_QWORD *)(v3 + 136) = 0;
    *(_QWORD *)(v3 + 152) = 0;
    *(_QWORD *)(v3 + 160) = 0;
    *(_BYTE *)(v3 + 168) = 0;
    *(_QWORD *)(v3 + 176) = a1;
    *(_QWORD *)(v3 + 184) = v3 + 200;
    *(_DWORD *)(v3 + 88) = 1065353216;
    *(_DWORD *)(v3 + 144) = 1065353216;
    sub_B3A000((__int64 *)(v3 + 184), v5, (__int64)&v5[v6]);
  }
  return v4;
}
