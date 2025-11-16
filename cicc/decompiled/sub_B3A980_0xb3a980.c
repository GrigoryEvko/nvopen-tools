// Function: sub_B3A980
// Address: 0xb3a980
//
__int64 __fastcall sub_B3A980(__int64 a1, __int64 a2, char a3)
{
  __int64 v5; // rax
  __int64 v6; // r12
  _BYTE *v7; // rsi
  __int64 v8; // rdx

  v5 = sub_22077B0(224);
  v6 = v5;
  if ( v5 )
  {
    v7 = *(_BYTE **)a2;
    *(_QWORD *)(v5 + 8) = 0;
    *(_QWORD *)(v5 + 16) = &unk_4F816F4;
    v8 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)(v5 + 56) = v5 + 104;
    *(_QWORD *)(v5 + 112) = v5 + 160;
    *(_QWORD *)v5 = off_49DA260;
    *(_DWORD *)(v5 + 24) = 4;
    *(_QWORD *)(v5 + 32) = 0;
    *(_QWORD *)(v5 + 40) = 0;
    *(_QWORD *)(v5 + 48) = 0;
    *(_QWORD *)(v5 + 64) = 1;
    *(_QWORD *)(v5 + 72) = 0;
    *(_QWORD *)(v5 + 80) = 0;
    *(_QWORD *)(v5 + 96) = 0;
    *(_QWORD *)(v5 + 104) = 0;
    *(_QWORD *)(v5 + 120) = 1;
    *(_QWORD *)(v5 + 128) = 0;
    *(_QWORD *)(v5 + 136) = 0;
    *(_QWORD *)(v5 + 152) = 0;
    *(_QWORD *)(v5 + 160) = 0;
    *(_BYTE *)(v5 + 168) = 0;
    *(_QWORD *)(v5 + 176) = a1;
    *(_QWORD *)(v5 + 184) = v5 + 200;
    *(_DWORD *)(v5 + 88) = 1065353216;
    *(_DWORD *)(v5 + 144) = 1065353216;
    sub_B3A000((__int64 *)(v5 + 184), v7, (__int64)&v7[v8]);
    *(_BYTE *)(v6 + 216) = a3;
  }
  return v6;
}
