// Function: sub_1A02300
// Address: 0x1a02300
//
__int64 sub_1A02300()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  __int64 v3; // rax

  v0 = sub_22077B0(920);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FB3CB8;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F5068;
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
    *(_QWORD *)(v0 + 160) = 0;
    *(_QWORD *)(v0 + 168) = 0;
    *(_QWORD *)(v0 + 176) = 0;
    *(_DWORD *)(v0 + 184) = 0;
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_DWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_QWORD *)(v0 + 232) = 0;
    *(_QWORD *)(v0 + 240) = 0;
    *(_DWORD *)(v0 + 248) = 0;
    *(_QWORD *)(v0 + 256) = 0;
    *(_QWORD *)(v0 + 264) = 0;
    *(_QWORD *)(v0 + 272) = 0;
    *(_QWORD *)(v0 + 280) = 0;
    *(_QWORD *)(v0 + 288) = 0;
    *(_QWORD *)(v0 + 296) = 0;
    *(_QWORD *)(v0 + 304) = 0;
    *(_QWORD *)(v0 + 312) = 0;
    *(_QWORD *)(v0 + 320) = 0;
    *(_QWORD *)(v0 + 328) = 0;
    sub_1A02210((__int64 *)(v0 + 256), 0);
    v2 = (_QWORD *)(v1 + 336);
    do
    {
      *v2 = 0;
      v2 += 4;
      *((_DWORD *)v2 - 2) = 0;
      *(v2 - 3) = 0;
      *((_DWORD *)v2 - 4) = 0;
      *((_DWORD *)v2 - 3) = 0;
    }
    while ( (_QWORD *)(v1 + 912) != v2 );
    v3 = sub_163A1D0();
    sub_1A01810(v3);
  }
  return v1;
}
