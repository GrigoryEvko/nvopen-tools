// Function: sub_1E12580
// Address: 0x1e12580
//
__int64 __fastcall sub_1E12580(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r12
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  __int64 v6; // rdi
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rax

  v2 = sub_22077B0(272);
  v3 = v2;
  if ( v2 )
  {
    *(_QWORD *)(v2 + 8) = 0;
    *(_QWORD *)(v2 + 16) = &unk_4FC64AC;
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
      a2 = 1;
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
      a2 = 1;
      sub_16BD1C0("Allocation failed", 1u);
      v5 = 0;
    }
    *v5 = 0;
    v6 = 8;
    *(_QWORD *)(v3 + 184) = v5;
    *(_QWORD *)(v3 + 192) = 1;
    *(_QWORD *)(v3 + 208) = 0;
    *(_QWORD *)(v3 + 216) = 0;
    *(_DWORD *)(v3 + 224) = 8;
    v7 = (_QWORD *)malloc(8u);
    if ( !v7 )
    {
      a2 = 1;
      v6 = (__int64)"Allocation failed";
      sub_16BD1C0("Allocation failed", 1u);
      v7 = 0;
    }
    *v7 = 0;
    *(_QWORD *)(v3 + 208) = v7;
    *(_QWORD *)(v3 + 216) = 1;
    *(_QWORD *)v3 = off_49FB858;
    v9 = sub_16BA580(v6, a2, v8);
    *(_QWORD *)(v3 + 248) = 0;
    *(_QWORD *)(v3 + 232) = v9;
    *(_QWORD *)(v3 + 240) = v3 + 256;
    *(_BYTE *)(v3 + 256) = 0;
  }
  return v3;
}
