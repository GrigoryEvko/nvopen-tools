// Function: sub_2C78D10
// Address: 0x2c78d10
//
__int64 sub_2C78D10()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rbx
  _BYTE *v3; // rax
  __int128 *v4; // rax

  v0 = sub_22077B0(0x168u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_5011094;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    v2 = v0 + 240;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A25050;
    *(_QWORD *)(v0 + 208) = v0 + 224;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_DWORD *)(v0 + 144) = 1065353216;
    *(_DWORD *)(v0 + 24) = 4;
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
    *(_DWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_BYTE *)(v0 + 224) = 0;
    *(_DWORD *)(v0 + 248) = 0;
    *(_BYTE *)(v0 + 280) = 0;
    *(_DWORD *)(v0 + 284) = 1;
    *(_QWORD *)(v0 + 288) = v0 + 208;
    *(_QWORD *)(v0 + 272) = 0;
    *(_QWORD *)(v0 + 240) = &unk_49DD210;
    *(_QWORD *)(v0 + 264) = 0;
    *(_QWORD *)(v0 + 256) = 0;
    sub_CB5980(v0 + 240, 0, 0, 0);
    v3 = *(_BYTE **)(v1 + 192);
    *(_QWORD *)(v1 + 296) = 0;
    *(_QWORD *)(v1 + 304) = 0;
    *(_QWORD *)(v1 + 312) = 0;
    *(_DWORD *)(v1 + 320) = 0;
    *(_QWORD *)(v1 + 328) = 0;
    *(_QWORD *)(v1 + 336) = 0;
    *(_QWORD *)(v1 + 344) = 0;
    *(_DWORD *)(v1 + 352) = 0;
    if ( v3 )
      *v3 = 1;
    if ( !*(_QWORD *)(v1 + 200) )
      *(_QWORD *)(v1 + 200) = v2;
    v4 = sub_BC2B00();
    sub_2C78C90((__int64)v4);
  }
  return v1;
}
