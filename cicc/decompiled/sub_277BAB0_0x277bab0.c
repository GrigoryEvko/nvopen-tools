// Function: sub_277BAB0
// Address: 0x277bab0
//
__int64 __fastcall sub_277BAB0(char a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  __int128 *v3; // rax
  __int64 v5; // rax
  __int128 *v6; // rax

  if ( a1 )
  {
    v1 = sub_22077B0(0xB0u);
    v2 = v1;
    if ( v1 )
    {
      *(_QWORD *)(v1 + 8) = 0;
      *(_QWORD *)(v1 + 16) = &unk_4FFB0EC;
      *(_QWORD *)(v1 + 56) = v1 + 104;
      *(_QWORD *)(v1 + 112) = v1 + 160;
      *(_DWORD *)(v1 + 24) = 2;
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
      *(_QWORD *)v1 = off_4A20B38;
      *(_DWORD *)(v1 + 88) = 1065353216;
      *(_DWORD *)(v1 + 144) = 1065353216;
      v3 = sub_BC2B00();
      sub_277BA30((__int64)v3);
    }
    return v2;
  }
  v5 = sub_22077B0(0xB0u);
  v2 = v5;
  if ( !v5 )
    return v2;
  *(_QWORD *)(v5 + 8) = 0;
  *(_QWORD *)(v5 + 16) = &unk_4FFB0F4;
  *(_QWORD *)(v5 + 56) = v5 + 104;
  *(_QWORD *)(v5 + 112) = v5 + 160;
  *(_DWORD *)(v5 + 24) = 2;
  *(_QWORD *)(v5 + 32) = 0;
  *(_QWORD *)(v5 + 40) = 0;
  *(_QWORD *)(v5 + 48) = 0;
  *(_QWORD *)(v5 + 64) = 1;
  *(_QWORD *)(v5 + 72) = 0;
  *(_QWORD *)(v5 + 80) = 0;
  *(_QWORD *)(v5 + 96) = 0;
  *(_QWORD *)(v5 + 104) = 0;
  *(_QWORD *)(v5 + 120) = 1;
  *(_QWORD *)(v5 + 128) = 0;
  *(_QWORD *)(v5 + 136) = 0;
  *(_QWORD *)(v5 + 152) = 0;
  *(_QWORD *)(v5 + 160) = 0;
  *(_BYTE *)(v5 + 168) = 0;
  *(_QWORD *)v5 = off_4A20A90;
  *(_DWORD *)(v5 + 88) = 1065353216;
  *(_DWORD *)(v5 + 144) = 1065353216;
  v6 = sub_BC2B00();
  sub_277B890((__int64)v6);
  return v2;
}
