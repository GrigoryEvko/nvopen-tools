// Function: sub_1F22ED0
// Address: 0x1f22ed0
//
__int64 sub_1F22ED0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // rax

  v0 = sub_22077B0(1568);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FCA83C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_DWORD *)(v0 + 24) = 3;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_DWORD *)(v0 + 64) = 0;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_DWORD *)(v0 + 112) = 0;
    *(_QWORD *)(v0 + 120) = 0;
    *(_QWORD *)(v0 + 144) = 0;
    *(_BYTE *)(v0 + 152) = 0;
    *(_QWORD *)v0 = &unk_49FB790;
    *(_QWORD *)(v0 + 160) = 0;
    *(_QWORD *)(v0 + 168) = 0;
    *(_DWORD *)(v0 + 176) = 8;
    v2 = (_QWORD *)malloc(8u);
    if ( !v2 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v2 = 0;
    }
    *v2 = 0;
    *(_QWORD *)(v1 + 160) = v2;
    *(_QWORD *)(v1 + 168) = 1;
    *(_QWORD *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 192) = 0;
    *(_DWORD *)(v1 + 200) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    *(_QWORD *)(v1 + 184) = v3;
    *(_QWORD *)(v1 + 192) = 1;
    *(_QWORD *)(v1 + 208) = 0;
    *(_QWORD *)(v1 + 216) = 0;
    *(_DWORD *)(v1 + 224) = 8;
    v4 = (_QWORD *)malloc(8u);
    if ( !v4 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = 0;
    }
    *(_QWORD *)(v1 + 208) = v4;
    *(_QWORD *)(v1 + 392) = v1 + 408;
    *v4 = 0;
    *(_QWORD *)(v1 + 400) = 0x1000000000LL;
    *(_QWORD *)(v1 + 544) = 0x1000000000LL;
    *(_QWORD *)v1 = off_49FE8F0;
    *(_QWORD *)(v1 + 1336) = v1 + 1352;
    *(_QWORD *)(v1 + 312) = v1 + 328;
    *(_QWORD *)(v1 + 1384) = v1 + 1400;
    *(_QWORD *)(v1 + 320) = 0x800000000LL;
    *(_QWORD *)(v1 + 536) = v1 + 552;
    *(_QWORD *)(v1 + 1344) = 0x400000000LL;
    *(_QWORD *)(v1 + 1432) = v1 + 1448;
    *(_QWORD *)(v1 + 1440) = 0x800000000LL;
    *(_QWORD *)(v1 + 216) = 1;
    *(_QWORD *)(v1 + 248) = 0;
    *(_QWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 264) = 0;
    *(_DWORD *)(v1 + 272) = 0;
    *(_QWORD *)(v1 + 280) = 0;
    *(_QWORD *)(v1 + 288) = 0;
    *(_QWORD *)(v1 + 296) = 0;
    *(_DWORD *)(v1 + 304) = 0;
    *(_QWORD *)(v1 + 1320) = 0;
    *(_QWORD *)(v1 + 1328) = 0;
    *(_QWORD *)(v1 + 1392) = 0;
    *(_QWORD *)(v1 + 1400) = 0;
    *(_QWORD *)(v1 + 1408) = 1;
    *(_QWORD *)(v1 + 1512) = 0;
    *(_QWORD *)(v1 + 1520) = 0;
    *(_DWORD *)(v1 + 1528) = 0;
    *(_QWORD *)(v1 + 1536) = 0;
    *(_QWORD *)(v1 + 1544) = 0;
    *(_DWORD *)(v1 + 1552) = 0;
    v5 = sub_163A1D0();
    sub_1F22DE0(v5);
  }
  return v1;
}
