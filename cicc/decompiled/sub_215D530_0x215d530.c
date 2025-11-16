// Function: sub_215D530
// Address: 0x215d530
//
__int64 sub_215D530()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rax
  _QWORD *i; // rdx
  _QWORD *v4; // rax
  _QWORD *j; // rdx

  v0 = sub_22077B0(320);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FD155C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_DWORD *)(v0 + 24) = 5;
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
    *(_QWORD *)v0 = off_4A01A88;
    *(_QWORD *)(v0 + 160) = 0;
    *(_DWORD *)(v0 + 184) = 128;
    v2 = (_QWORD *)sub_22077B0(6144);
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 168) = v2;
    for ( i = &v2[6 * *(unsigned int *)(v1 + 184)]; i != v2; v2 += 6 )
    {
      if ( v2 )
      {
        v2[2] = 0;
        v2[3] = -8;
        *v2 = &unk_49F8530;
        v2[1] = 2;
        v2[4] = 0;
      }
    }
    *(_BYTE *)(v1 + 224) = 0;
    *(_BYTE *)(v1 + 233) = 1;
    *(_QWORD *)(v1 + 240) = 0;
    *(_DWORD *)(v1 + 264) = 128;
    v4 = (_QWORD *)sub_22077B0(6144);
    *(_QWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 248) = v4;
    for ( j = &v4[6 * *(unsigned int *)(v1 + 264)]; j != v4; v4 += 6 )
    {
      if ( v4 )
      {
        v4[2] = 0;
        v4[3] = -8;
        *v4 = &unk_4A01B30;
        v4[1] = 2;
        v4[4] = 0;
      }
    }
    *(_BYTE *)(v1 + 304) = 0;
    *(_BYTE *)(v1 + 313) = 1;
  }
  return v1;
}
