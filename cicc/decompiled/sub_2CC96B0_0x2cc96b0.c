// Function: sub_2CC96B0
// Address: 0x2cc96b0
//
__int64 sub_2CC96B0()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int128 *v2; // rax

  v0 = sub_22077B0(0xB0u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_DWORD *)(v0 + 24) = 2;
    *(_QWORD *)(v0 + 16) = &unk_50139EC;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)(v0 + 32) = 0;
    *(_DWORD *)(v0 + 88) = 1065353216;
    *(_QWORD *)v0 = &unk_4A255C8;
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
    *(_WORD *)(v0 + 168) = 256;
    *(_DWORD *)(v0 + 144) = 1065353216;
    v2 = sub_BC2B00();
    sub_2CC9630((__int64)v2);
  }
  return v1;
}
