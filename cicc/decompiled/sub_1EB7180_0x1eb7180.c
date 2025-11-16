// Function: sub_1EB7180
// Address: 0x1eb7180
//
__int64 sub_1EB7180()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax

  v0 = sub_22077B0(1096);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC9194;
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
    *v4 = 0;
    *(_QWORD *)(v1 + 208) = v4;
    *(_QWORD *)v1 = off_49FD9B0;
    *(_QWORD *)(v1 + 216) = 1;
    sub_1ED72C0(v1 + 264);
    *(_QWORD *)(v1 + 368) = v1 + 384;
    *(_QWORD *)(v1 + 672) = v1 + 688;
    *(_QWORD *)(v1 + 392) = v1 + 408;
    *(_QWORD *)(v1 + 752) = v1 + 768;
    *(_QWORD *)(v1 + 376) = 0;
    *(_DWORD *)(v1 + 384) = -1;
    *(_QWORD *)(v1 + 400) = 0x800000000LL;
    *(_QWORD *)(v1 + 600) = 0;
    *(_DWORD *)(v1 + 608) = 0;
    *(_QWORD *)(v1 + 616) = 0;
    *(_QWORD *)(v1 + 624) = 0;
    *(_QWORD *)(v1 + 632) = 0;
    *(_DWORD *)(v1 + 640) = 0;
    *(_QWORD *)(v1 + 648) = 0;
    *(_QWORD *)(v1 + 656) = 0;
    *(_QWORD *)(v1 + 664) = 0;
    *(_QWORD *)(v1 + 680) = 0x1000000000LL;
    *(_QWORD *)(v1 + 760) = 0x2000000000LL;
    *(_QWORD *)(v1 + 1024) = v1 + 1040;
    *(_QWORD *)(v1 + 1032) = 0x800000000LL;
    *(_QWORD *)(v1 + 1072) = 0;
    *(_DWORD *)(v1 + 1080) = 0;
    *(_BYTE *)(v1 + 1088) = 0;
  }
  return v1;
}
