// Function: sub_217DCE0
// Address: 0x217dce0
//
__int64 sub_217DCE0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rax

  v0 = sub_22077B0(384);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4CD4B10;
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
    sub_1BFC1A0(v1 + 184, 8, 0);
    *(_QWORD *)(v1 + 208) = 0;
    *(_QWORD *)(v1 + 216) = 0;
    *(_DWORD *)(v1 + 224) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    *(_QWORD *)(v1 + 208) = v3;
    *(_QWORD *)(v1 + 216) = 1;
    *(_QWORD *)v1 = off_4A031D0;
    *(_QWORD *)(v1 + 240) = 0;
    *(_QWORD *)(v1 + 288) = 0;
    *(_QWORD *)(v1 + 296) = 0;
    *(_QWORD *)(v1 + 304) = 0;
    *(_DWORD *)(v1 + 312) = 0;
    *(_QWORD *)(v1 + 320) = 0;
    *(_QWORD *)(v1 + 328) = 0;
    *(_QWORD *)(v1 + 336) = 0;
    *(_DWORD *)(v1 + 344) = 0;
    *(_QWORD *)(v1 + 352) = 0;
    *(_QWORD *)(v1 + 360) = 0;
    *(_QWORD *)(v1 + 368) = 0;
    *(_DWORD *)(v1 + 376) = 0;
    v4 = sub_163A1D0();
    sub_217DBF0(v4);
  }
  return v1;
}
