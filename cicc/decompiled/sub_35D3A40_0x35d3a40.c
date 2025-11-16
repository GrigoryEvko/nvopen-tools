// Function: sub_35D3A40
// Address: 0x35d3a40
//
__int64 sub_35D3A40()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0x110u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_5040114;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A3A988;
    *(_QWORD *)(v0 + 216) = v0 + 240;
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
    *(_QWORD *)(v0 + 200) = 0;
    *(_QWORD *)(v0 + 208) = 0;
    *(_QWORD *)(v0 + 224) = 0;
    *(_QWORD *)(v0 + 232) = 8;
    *(_QWORD *)(v0 + 256) = 0;
    *(_DWORD *)(v0 + 264) = 0;
    v2 = sub_BC2B00();
    sub_35D39C0((__int64)v2);
  }
  return v1;
}
