// Function: sub_39B99F0
// Address: 0x39b99f0
//
__int64 __fastcall sub_39B99F0(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // r12
  _QWORD *v3; // rax
  _QWORD *v4; // rax
  _QWORD *v5; // rax

  v1 = sub_22077B0(0x110u);
  v2 = v1;
  if ( v1 )
  {
    *(_QWORD *)(v1 + 8) = 0;
    *(_QWORD *)(v1 + 16) = &unk_5057608;
    *(_QWORD *)(v1 + 80) = v1 + 64;
    *(_QWORD *)(v1 + 88) = v1 + 64;
    *(_QWORD *)(v1 + 128) = v1 + 112;
    *(_QWORD *)(v1 + 136) = v1 + 112;
    *(_DWORD *)(v1 + 24) = 3;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_DWORD *)(v1 + 64) = 0;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_DWORD *)(v1 + 112) = 0;
    *(_QWORD *)(v1 + 120) = 0;
    *(_QWORD *)(v1 + 144) = 0;
    *(_BYTE *)(v1 + 152) = 0;
    *(_QWORD *)v1 = &unk_49FB790;
    *(_QWORD *)(v1 + 160) = 0;
    *(_QWORD *)(v1 + 168) = 0;
    *(_DWORD *)(v1 + 176) = 8;
    v3 = (_QWORD *)malloc(8u);
    if ( !v3 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v3 = 0;
    }
    *v3 = 0;
    *(_QWORD *)(v2 + 160) = v3;
    *(_QWORD *)(v2 + 168) = 1;
    *(_QWORD *)(v2 + 184) = 0;
    *(_QWORD *)(v2 + 192) = 0;
    *(_DWORD *)(v2 + 200) = 8;
    v4 = (_QWORD *)malloc(8u);
    if ( !v4 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = 0;
    }
    *v4 = 0;
    *(_QWORD *)(v2 + 184) = v4;
    *(_QWORD *)(v2 + 192) = 1;
    *(_QWORD *)(v2 + 208) = 0;
    *(_QWORD *)(v2 + 216) = 0;
    *(_DWORD *)(v2 + 224) = 8;
    v5 = (_QWORD *)malloc(8u);
    if ( !v5 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = 0;
    }
    *(_QWORD *)(v2 + 208) = v5;
    *v5 = 0;
    *(_QWORD *)v2 = off_4A40370;
    *(_QWORD *)(v2 + 216) = 1;
    *(_QWORD *)(v2 + 232) = a1;
    *(_QWORD *)(v2 + 240) = v2 + 256;
    *(_QWORD *)(v2 + 248) = 0;
    *(_BYTE *)(v2 + 256) = 0;
  }
  return v2;
}
