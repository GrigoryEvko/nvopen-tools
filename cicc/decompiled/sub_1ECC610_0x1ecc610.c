// Function: sub_1ECC610
// Address: 0x1ecc610
//
__int64 __fastcall sub_1ECC610(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  _QWORD *v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax

  v1 = sub_22077B0(632);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_4FC9B80;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_DWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_QWORD *)v1 = &unk_49FB790;
    *(_QWORD *)(v1 + 160) = 0;
    *(_QWORD *)(v1 + 168) = 0;
    *(_DWORD *)(v1 + 176) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    *(_QWORD *)(v2 + 160) = v3;
    *(_QWORD *)(v2 + 168) = 1;
    sub_1BFC1A0(v2 + 184, 8, 0);
    sub_1BFC1A0(v2 + 208, 8, 0);
    *(_QWORD *)(v2 + 232) = a1;
    *(_QWORD *)v2 = off_49FDDD0;
    *(_QWORD *)(v2 + 264) = v2 + 248;
    *(_QWORD *)(v2 + 272) = v2 + 248;
    *(_QWORD *)(v2 + 312) = v2 + 296;
    *(_QWORD *)(v2 + 320) = v2 + 296;
    *(_DWORD *)(v2 + 248) = 0;
    *(_QWORD *)(v2 + 256) = 0;
    *(_QWORD *)(v2 + 280) = 0;
    *(_DWORD *)(v2 + 296) = 0;
    *(_QWORD *)(v2 + 304) = 0;
    *(_QWORD *)(v2 + 328) = 0;
    *(_QWORD *)(v2 + 336) = 0;
    *(_QWORD *)(v2 + 344) = v2 + 376;
    *(_QWORD *)(v2 + 352) = v2 + 376;
    *(_QWORD *)(v2 + 360) = 32;
    *(_DWORD *)(v2 + 368) = 0;
    v4 = sub_163A1D0();
    sub_1F10320(v4);
    v5 = sub_163A1D0();
    sub_1DB9DF0(v5);
    v6 = sub_163A1D0();
    sub_1DC9950(v6);
    v7 = sub_163A1D0();
    sub_1F5BA80(v7);
  }
  return v2;
}
