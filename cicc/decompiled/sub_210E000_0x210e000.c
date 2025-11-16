// Function: sub_210E000
// Address: 0x210e000
//
__int64 __fastcall sub_210E000(char a1, char a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rax

  v2 = sub_22077B0(240);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4FCFB28;
    *(_QWORD *)(v2 + 80) = v2 + 64;
    *(_QWORD *)(v2 + 88) = v2 + 64;
    *(_QWORD *)(v2 + 128) = v2 + 112;
    *(_QWORD *)(v2 + 136) = v2 + 112;
    *(_DWORD *)(v2 + 24) = 3;
    *(_QWORD *)(v2 + 32) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 48) = 0;
    *(_DWORD *)(v2 + 64) = 0;
    *(_QWORD *)(v2 + 72) = 0;
    *(_QWORD *)(v2 + 96) = 0;
    *(_DWORD *)(v2 + 112) = 0;
    *(_QWORD *)(v2 + 120) = 0;
    *(_QWORD *)(v2 + 144) = 0;
    *(_BYTE *)(v2 + 152) = 0;
    *(_QWORD *)v2 = &unk_49FB790;
    *(_QWORD *)(v2 + 160) = 0;
    *(_QWORD *)(v2 + 168) = 0;
    *(_DWORD *)(v2 + 176) = 8;
    v4 = (_QWORD *)malloc(8u);
    if ( !v4 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v4 = 0;
    }
    *v4 = 0;
    *(_QWORD *)(v3 + 160) = v4;
    *(_QWORD *)(v3 + 168) = 1;
    *(_QWORD *)(v3 + 184) = 0;
    *(_QWORD *)(v3 + 192) = 0;
    *(_DWORD *)(v3 + 200) = 8;
    v5 = (_QWORD *)malloc(8u);
    if ( !v5 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = 0;
    }
    *v5 = 0;
    *(_QWORD *)(v3 + 184) = v5;
    *(_QWORD *)(v3 + 192) = 1;
    *(_QWORD *)(v3 + 208) = 0;
    *(_QWORD *)(v3 + 216) = 0;
    *(_DWORD *)(v3 + 224) = 8;
    v6 = (_QWORD *)malloc(8u);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = 0;
    }
    *(_QWORD *)(v3 + 208) = v6;
    *v6 = 0;
    *(_QWORD *)(v3 + 216) = 1;
    *(_QWORD *)v3 = off_4A00F80;
    *(_BYTE *)(v3 + 232) = a1;
    *(_BYTE *)(v3 + 233) = a2;
  }
  return v3;
}
