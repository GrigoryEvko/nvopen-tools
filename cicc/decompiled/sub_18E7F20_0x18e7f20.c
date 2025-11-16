// Function: sub_18E7F20
// Address: 0x18e7f20
//
__int64 sub_18E7F20()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _QWORD *v2; // rdx
  _QWORD *v3; // rax
  __int64 v4; // rax

  v0 = sub_22077B0(5368);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = (_QWORD *)(v0 + 296);
    *(_QWORD *)(v0 + 16) = &unk_4FAE19C;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
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
    *(_QWORD *)(v0 + 192) = 0;
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 224) = 1;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F2D08;
    v3 = (_QWORD *)(v0 + 232);
    do
    {
      if ( v3 )
        *v3 = -8;
      v3 += 2;
    }
    while ( v2 != v3 );
    *(_QWORD *)(v1 + 296) = v1 + 312;
    *(_QWORD *)(v1 + 304) = 0x800000000LL;
    v4 = sub_163A1D0();
    sub_18E7E30(v4);
  }
  return v1;
}
