// Function: sub_21F9A00
// Address: 0x21f9a00
//
__int64 sub_21F9A00()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  __int64 v5; // rax

  v0 = sub_22077B0(512);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FD42F8;
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
    *(_QWORD *)(v1 + 216) = 1;
    *(_QWORD *)v1 = off_4A04188;
    *(_QWORD *)(v1 + 232) = 0;
    *(_QWORD *)(v1 + 240) = 0;
    *(_QWORD *)(v1 + 248) = 0;
    *(_DWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 264) = 0;
    *(_QWORD *)(v1 + 272) = 0;
    *(_QWORD *)(v1 + 280) = 0;
    *(_DWORD *)(v1 + 288) = 0;
    *(_QWORD *)(v1 + 296) = 0;
    *(_QWORD *)(v1 + 304) = 0;
    *(_QWORD *)(v1 + 312) = 0;
    *(_DWORD *)(v1 + 320) = 0;
    *(_QWORD *)(v1 + 328) = 0;
    *(_QWORD *)(v1 + 336) = 0;
    *(_QWORD *)(v1 + 344) = 0;
    *(_DWORD *)(v1 + 352) = 0;
    *(_QWORD *)(v1 + 360) = 0;
    *(_QWORD *)(v1 + 368) = 0;
    *(_QWORD *)(v1 + 376) = 0;
    *(_DWORD *)(v1 + 384) = 0;
    *(_QWORD *)(v1 + 392) = 0;
    *(_QWORD *)(v1 + 400) = 0;
    *(_QWORD *)(v1 + 408) = 0;
    *(_DWORD *)(v1 + 416) = 0;
    *(_QWORD *)(v1 + 440) = 0;
    *(_QWORD *)(v1 + 448) = 0;
    *(_QWORD *)(v1 + 456) = 0;
    *(_DWORD *)(v1 + 464) = 0;
    v5 = sub_163A1D0();
    sub_21F9920(v5);
  }
  return v1;
}
