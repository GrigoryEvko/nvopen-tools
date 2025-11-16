// Function: sub_1892EF0
// Address: 0x1892ef0
//
__int64 sub_1892EF0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 v2; // rbx
  _QWORD *v3; // rax
  _QWORD *i; // rdx
  _QWORD *v5; // rax
  __int64 v6; // rdx
  _QWORD *j; // rdx
  __int64 v8; // rax

  v0 = sub_22077B0(400);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = v0 + 160;
    *(_QWORD *)(v0 + 16) = &unk_4FAC5CC;
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
    *(_QWORD *)v0 = off_49F1CE8;
    *(_QWORD *)(v0 + 160) = 0;
    *(_DWORD *)(v0 + 184) = 128;
    v3 = (_QWORD *)sub_22077B0(6144);
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 168) = v3;
    for ( i = &v3[6 * *(unsigned int *)(v1 + 184)]; i != v3; v3 += 6 )
    {
      if ( v3 )
      {
        v3[2] = 0;
        v3[3] = -8;
        *v3 = &unk_49F1DB8;
        v3[1] = 2;
        v3[4] = 0;
      }
    }
    *(_BYTE *)(v1 + 224) = 0;
    *(_BYTE *)(v1 + 233) = 1;
    *(_QWORD *)(v1 + 240) = 0;
    *(_QWORD *)(v1 + 248) = 0;
    *(_QWORD *)(v1 + 256) = 0;
    *(_QWORD *)(v1 + 264) = 0;
    *(_QWORD *)(v1 + 272) = v2;
    *(_DWORD *)(v1 + 280) = 0;
    *(_QWORD *)(v1 + 288) = 0;
    *(_QWORD *)(v1 + 296) = v1 + 280;
    *(_QWORD *)(v1 + 304) = v1 + 280;
    *(_QWORD *)(v1 + 312) = 0;
    *(_QWORD *)(v1 + 320) = 0;
    *(_DWORD *)(v1 + 344) = 128;
    v5 = (_QWORD *)sub_22077B0(6144);
    v6 = *(unsigned int *)(v1 + 344);
    *(_QWORD *)(v1 + 336) = 0;
    *(_QWORD *)(v1 + 328) = v5;
    for ( j = &v5[6 * v6]; j != v5; v5 += 6 )
    {
      if ( v5 )
      {
        v5[2] = 0;
        v5[3] = -8;
        *v5 = off_49F1D90;
        v5[1] = 2;
        v5[4] = 0;
      }
    }
    *(_BYTE *)(v1 + 384) = 0;
    *(_BYTE *)(v1 + 393) = 1;
    v8 = sub_163A1D0();
    sub_1892B20(v8);
  }
  return v1;
}
