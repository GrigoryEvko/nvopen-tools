// Function: sub_349D300
// Address: 0x349d300
//
__int64 sub_349D300()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int128 *v2; // rax
  __int64 v3; // rax
  __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rdi

  v0 = sub_22077B0(0x160u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_503A0EC;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A375B0;
    *(_QWORD *)(v0 + 224) = v0 + 240;
    *(_QWORD *)(v0 + 232) = 0x100000000LL;
    *(_QWORD *)(v0 + 248) = v0 + 264;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 32) = 0;
    *(_QWORD *)(v0 + 40) = 0;
    *(_QWORD *)(v0 + 48) = 0;
    *(_QWORD *)(v0 + 64) = 1;
    *(_QWORD *)(v0 + 72) = 0;
    *(_QWORD *)(v0 + 80) = 0;
    *(_QWORD *)(v0 + 96) = 0;
    *(_QWORD *)(v0 + 104) = 0;
    *(_QWORD *)(v0 + 120) = 1;
    *(_QWORD *)(v0 + 128) = 0;
    *(_QWORD *)(v0 + 136) = 0;
    *(_QWORD *)(v0 + 152) = 0;
    *(_QWORD *)(v0 + 160) = 0;
    *(_BYTE *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_QWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 256) = 0x600000000LL;
    *(_QWORD *)(v0 + 320) = 0;
    *(_QWORD *)(v0 + 328) = 0;
    *(_BYTE *)(v0 + 336) = 0;
    *(_QWORD *)(v0 + 340) = 0;
    v2 = sub_BC2B00();
    sub_349D280((__int64)v2);
    v3 = sub_37BC0F0();
    v4 = *(_QWORD *)(v1 + 200);
    *(_QWORD *)(v1 + 200) = v3;
    if ( v4 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 16LL))(v4);
    v5 = sub_34A0420();
    v6 = *(_QWORD *)(v1 + 208);
    *(_QWORD *)(v1 + 208) = v5;
    if ( v6 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v6 + 16LL))(v6);
  }
  return v1;
}
