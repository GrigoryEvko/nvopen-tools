// Function: sub_1D82840
// Address: 0x1d82840
//
__int64 sub_1D82840()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax

  v0 = sub_22077B0(1064);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC3344;
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
    *v4 = 0;
    *(_QWORD *)v1 = off_49FA098;
    *(_QWORD *)(v1 + 688) = v1 + 704;
    *(_QWORD *)(v1 + 416) = v1 + 432;
    *(_QWORD *)(v1 + 872) = v1 + 904;
    *(_QWORD *)(v1 + 880) = v1 + 904;
    *(_QWORD *)(v1 + 216) = 1;
    *(_QWORD *)(v1 + 424) = 0x800000000LL;
    *(_QWORD *)(v1 + 696) = 0x400000000LL;
    *(_QWORD *)(v1 + 864) = 0;
    *(_QWORD *)(v1 + 888) = 8;
    *(_DWORD *)(v1 + 896) = 0;
    *(_QWORD *)(v1 + 968) = 0;
    *(_QWORD *)(v1 + 976) = 0;
    *(_DWORD *)(v1 + 984) = 0;
    *(_QWORD *)(v1 + 992) = v1 + 1008;
    *(_QWORD *)(v1 + 1000) = 0x800000000LL;
    *(_QWORD *)(v1 + 1040) = 0;
    *(_DWORD *)(v1 + 1048) = 0;
    *(_QWORD *)(v1 + 1056) = 0;
  }
  return v1;
}
