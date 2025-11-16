// Function: sub_35B99A0
// Address: 0x35b99a0
//
__int64 __fastcall sub_35B99A0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int128 *v3; // rax
  __int128 *v4; // rax
  __int128 *v5; // rax
  __int128 *v6; // rax

  v1 = sub_22077B0(0x250u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_503FE10;
    *(_QWORD *)(v1 + 56) = v1 + 104;
    *(_QWORD *)(v1 + 112) = v1 + 160;
    *(_QWORD *)v1 = off_4A3A2E0;
    *(_QWORD *)(v1 + 232) = v1 + 216;
    *(_QWORD *)(v1 + 240) = v1 + 216;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    *(_DWORD *)(v1 + 24) = 2;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 192) = 0;
    *(_QWORD *)(v1 + 200) = a1;
    *(_DWORD *)(v1 + 216) = 0;
    *(_QWORD *)(v1 + 224) = 0;
    *(_QWORD *)(v1 + 248) = 0;
    *(_DWORD *)(v1 + 264) = 0;
    *(_QWORD *)(v1 + 280) = v1 + 264;
    *(_QWORD *)(v1 + 288) = v1 + 264;
    *(_QWORD *)(v1 + 272) = 0;
    *(_QWORD *)(v1 + 296) = 0;
    *(_QWORD *)(v1 + 304) = 0;
    *(_QWORD *)(v1 + 312) = v1 + 336;
    *(_QWORD *)(v1 + 320) = 32;
    *(_DWORD *)(v1 + 328) = 0;
    *(_BYTE *)(v1 + 332) = 1;
    v3 = sub_BC2B00();
    sub_2FACF50((__int64)v3);
    v4 = sub_BC2B00();
    sub_2E10620((__int64)v4);
    v5 = sub_BC2B00();
    sub_2E22F70((__int64)v5);
    v6 = sub_BC2B00();
    sub_300B990((__int64)v6);
  }
  return v2;
}
