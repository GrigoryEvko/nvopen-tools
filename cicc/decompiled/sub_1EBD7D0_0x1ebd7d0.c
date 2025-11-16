// Function: sub_1EBD7D0
// Address: 0x1ebd7d0
//
__int64 __fastcall sub_1EBD7D0(__int64 a1)
{
  _QWORD *v1; // rax
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  _QWORD *v6; // rax

  *(_QWORD *)(a1 + 16) = &unk_4FC91D8;
  *(_QWORD *)(a1 + 80) = a1 + 64;
  *(_QWORD *)(a1 + 88) = a1 + 64;
  *(_QWORD *)(a1 + 128) = a1 + 112;
  *(_QWORD *)(a1 + 136) = a1 + 112;
  *(_QWORD *)(a1 + 8) = 0;
  *(_DWORD *)(a1 + 24) = 3;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_DWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_BYTE *)(a1 + 152) = 0;
  *(_QWORD *)a1 = &unk_49FB790;
  *(_QWORD *)(a1 + 160) = 0;
  *(_QWORD *)(a1 + 168) = 0;
  *(_DWORD *)(a1 + 176) = 8;
  v1 = (_QWORD *)malloc(8u);
  if ( !v1 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v1 = 0;
  }
  *(_QWORD *)(a1 + 160) = v1;
  *(_QWORD *)(a1 + 168) = 1;
  *v1 = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 200) = 8;
  v2 = (_QWORD *)malloc(8u);
  if ( !v2 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v2 = 0;
  }
  *(_QWORD *)(a1 + 184) = v2;
  *(_QWORD *)(a1 + 192) = 1;
  *v2 = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_DWORD *)(a1 + 224) = 8;
  v3 = (_QWORD *)malloc(8u);
  if ( !v3 )
  {
    sub_16BD1C0("Allocation failed", 1u);
    v3 = 0;
  }
  *(_QWORD *)(a1 + 208) = v3;
  *v3 = 0;
  *(_QWORD *)(a1 + 216) = 1;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 232) = &unk_4A00E78;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 272) = 0;
  sub_1ED72C0(a1 + 280);
  *(_QWORD *)(a1 + 376) = 0;
  *(_QWORD *)(a1 + 384) = a1 + 416;
  *(_QWORD *)(a1 + 392) = a1 + 416;
  *(_QWORD *)a1 = off_49FDB68;
  *(_QWORD *)(a1 + 232) = &unk_49FDC70;
  *(_QWORD *)(a1 + 672) = &unk_49FDCC8;
  *(_QWORD *)(a1 + 400) = 32;
  *(_DWORD *)(a1 + 408) = 0;
  sub_1ED72C0(a1 + 704);
  *(_QWORD *)(a1 + 872) = 0;
  *(_QWORD *)(a1 + 880) = 0;
  *(_QWORD *)(a1 + 888) = 0;
  *(_QWORD *)(a1 + 896) = 0;
  *(_QWORD *)(a1 + 928) = 0;
  *(_QWORD *)(a1 + 936) = 0;
  *(_QWORD *)(a1 + 952) = 0;
  *(_QWORD *)(a1 + 960) = 0;
  *(_QWORD *)(a1 + 968) = 0;
  *(_DWORD *)(a1 + 976) = 0;
  *(_QWORD *)(a1 + 984) = 0;
  *(_QWORD *)(a1 + 992) = 0;
  *(_QWORD *)(a1 + 1000) = 0;
  *(_QWORD *)(a1 + 1008) = 0;
  *(_QWORD *)(a1 + 1016) = 0;
  *(_QWORD *)(a1 + 1024) = 0;
  *(_QWORD *)(a1 + 1032) = 0;
  *(_DWORD *)(a1 + 1040) = 0;
  *(_QWORD *)(a1 + 920) = a1 + 936;
  v4 = a1 + 1048;
  do
  {
    *(_DWORD *)v4 = 0;
    *(_QWORD *)(v4 + 48) = v4 + 64;
    v5 = v4 + 528;
    v4 += 720;
    *(_DWORD *)(v4 - 716) = 0;
    *(_DWORD *)(v4 - 712) = 0;
    *(_QWORD *)(v4 - 696) = 0;
    *(_QWORD *)(v4 - 688) = 0;
    *(_QWORD *)(v4 - 680) = 0;
    *(_DWORD *)(v4 - 664) = 0;
    *(_DWORD *)(v4 - 660) = 4;
    *(_QWORD *)(v4 - 208) = v5;
    *(_DWORD *)(v4 - 200) = 0;
    *(_DWORD *)(v4 - 196) = 8;
  }
  while ( v4 != a1 + 24088 );
  *(_QWORD *)(a1 + 27400) = 0;
  *(_QWORD *)(a1 + 24088) = a1 + 24104;
  *(_QWORD *)(a1 + 24096) = 0x800000000LL;
  *(_QWORD *)(a1 + 27416) = 0;
  *(_QWORD *)(a1 + 27424) = 1;
  *(_QWORD *)(a1 + 24168) = a1 + 24184;
  *(_QWORD *)(a1 + 24176) = 0x2000000000LL;
  *(_QWORD *)(a1 + 27256) = a1 + 27272;
  *(_QWORD *)(a1 + 27264) = 0x2000000000LL;
  v6 = (_QWORD *)(a1 + 27432);
  do
  {
    if ( v6 )
      *v6 = -8;
    ++v6;
  }
  while ( (_QWORD *)(a1 + 27496) != v6 );
  *(_QWORD *)(a1 + 27496) = a1 + 27512;
  *(_QWORD *)(a1 + 27504) = 0x800000000LL;
  return 0x800000000LL;
}
