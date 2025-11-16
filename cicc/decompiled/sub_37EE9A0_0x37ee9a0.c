// Function: sub_37EE9A0
// Address: 0x37ee9a0
//
__int64 sub_37EE9A0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  _DWORD *v2; // rdx
  _DWORD *v3; // rax
  __int128 *v4; // rax

  v0 = sub_22077B0(0x230u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = (_DWORD *)(v0 + 560);
    *(_QWORD *)(v0 + 16) = &unk_5051344;
    *(_QWORD *)(v0 + 56) = v0 + 104;
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
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 216) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_QWORD *)(v0 + 232) = 1;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A3D790;
    v3 = (_DWORD *)(v0 + 240);
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    do
    {
      if ( v3 )
        *v3 = -1;
      v3 += 5;
    }
    while ( v3 != v2 );
    v4 = sub_BC2B00();
    sub_37EE920((__int64)v4);
  }
  return v1;
}
