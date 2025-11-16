// Function: sub_1EAC700
// Address: 0x1eac700
//
__int64 sub_1EAC700()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rax

  v0 = sub_22077B0(544);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FC9074;
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
    *(_QWORD *)(v1 + 216) = 1;
    *(_QWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 264) = 1;
    *v4 = 0;
    *(_QWORD *)v1 = off_49FD5B0;
    v5 = (_QWORD *)(v1 + 272);
    do
    {
      if ( v5 )
        *v5 = -8;
      ++v5;
    }
    while ( (_QWORD *)(v1 + 400) != v5 );
    *(_QWORD *)(v1 + 400) = v1 + 416;
    *(_QWORD *)(v1 + 408) = 0x1000000000LL;
    v6 = sub_163A1D0();
    sub_1EAC620(v6);
  }
  return v1;
}
