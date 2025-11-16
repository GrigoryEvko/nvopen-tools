// Function: sub_24FB600
// Address: 0x24fb600
//
__int64 __fastcall sub_24FB600(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int128 *v3; // rax

  v1 = sub_22077B0(0xB0u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_4FEE4AC;
    *(_QWORD *)(v1 + 56) = v1 + 104;
    *(_QWORD *)(v1 + 112) = v1 + 160;
    *(_DWORD *)(v1 + 24) = 4;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)v1 = off_4A16B58;
    *(_BYTE *)(v1 + 169) = a1;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    v3 = sub_BC2B00();
    sub_24FB460((__int64)v3);
  }
  return v2;
}
