// Function: sub_28EE4F0
// Address: 0x28ee4f0
//
__int64 sub_28EE4F0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  __int128 *v3; // rax

  v0 = sub_22077B0(0x3A8u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_5004BAC;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A21D00;
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
    *(_DWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_DWORD *)(v0 + 232) = 0;
    *(_QWORD *)(v0 + 240) = 0;
    *(_QWORD *)(v0 + 248) = 0;
    *(_QWORD *)(v0 + 256) = 0;
    *(_DWORD *)(v0 + 264) = 0;
    *(_QWORD *)(v0 + 272) = 0;
    *(_QWORD *)(v0 + 280) = 0;
    *(_QWORD *)(v0 + 288) = 0;
    *(_QWORD *)(v0 + 296) = 0;
    *(_QWORD *)(v0 + 304) = 0;
    *(_QWORD *)(v0 + 312) = 0;
    *(_QWORD *)(v0 + 320) = 0;
    *(_QWORD *)(v0 + 328) = 0;
    *(_QWORD *)(v0 + 336) = 0;
    *(_QWORD *)(v0 + 344) = 0;
    sub_2350260((__int64 *)(v0 + 272), 0);
    v2 = (_QWORD *)(v1 + 352);
    do
    {
      *v2 = 0;
      v2 += 4;
      *((_DWORD *)v2 - 2) = 0;
      *(v2 - 3) = 0;
      *((_DWORD *)v2 - 4) = 0;
      *((_DWORD *)v2 - 3) = 0;
    }
    while ( v2 != (_QWORD *)(v1 + 928) );
    v3 = sub_BC2B00();
    sub_28ED680((__int64)v3);
  }
  return v1;
}
