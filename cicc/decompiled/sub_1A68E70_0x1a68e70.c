// Function: sub_1A68E70
// Address: 0x1a68e70
//
__int64 sub_1A68E70()
{
  __int64 v0; // rax
  __int64 v1; // r12

  v0 = sub_22077B0(176);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    *(_QWORD *)(v0 + 16) = &unk_4FB4ACC;
    *(_QWORD *)(v0 + 80) = v0 + 64;
    *(_QWORD *)(v0 + 88) = v0 + 64;
    *(_QWORD *)(v0 + 128) = v0 + 112;
    *(_QWORD *)(v0 + 136) = v0 + 112;
    *(_QWORD *)v0 = off_49F5678;
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
    *(_WORD *)(v0 + 152) = 256;
    sub_1A68C70(v0 + 160, 1);
  }
  return v1;
}
