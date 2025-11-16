// Function: sub_1E128C0
// Address: 0x1e128c0
//
__int64 __fastcall sub_1E128C0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _BYTE *v8; // r14
  _BYTE *v9; // rdi
  size_t v10; // r13
  __int64 v12; // rax
  size_t v13[5]; // [rsp+18h] [rbp-28h] BYREF

  v3 = sub_22077B0(272);
  v4 = v3;
  if ( v3 )
  {
    *(_QWORD *)(v3 + 8) = 0;
    *(_QWORD *)(v3 + 16) = &unk_4FC64AC;
    *(_QWORD *)(v3 + 80) = v3 + 64;
    *(_QWORD *)(v3 + 88) = v3 + 64;
    *(_QWORD *)(v3 + 128) = v3 + 112;
    *(_QWORD *)(v3 + 136) = v3 + 112;
    *(_DWORD *)(v3 + 24) = 3;
    *(_QWORD *)(v3 + 32) = 0;
    *(_QWORD *)(v3 + 40) = 0;
    *(_QWORD *)(v3 + 48) = 0;
    *(_DWORD *)(v3 + 64) = 0;
    *(_QWORD *)(v3 + 72) = 0;
    *(_QWORD *)(v3 + 96) = 0;
    *(_DWORD *)(v3 + 112) = 0;
    *(_QWORD *)(v3 + 120) = 0;
    *(_QWORD *)(v3 + 144) = 0;
    *(_BYTE *)(v3 + 152) = 0;
    *(_QWORD *)v3 = &unk_49FB790;
    *(_QWORD *)(v3 + 160) = 0;
    *(_QWORD *)(v3 + 168) = 0;
    *(_DWORD *)(v3 + 176) = 8;
    v5 = (_QWORD *)malloc(8u);
    if ( !v5 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v5 = 0;
    }
    *v5 = 0;
    *(_QWORD *)(v4 + 160) = v5;
    *(_QWORD *)(v4 + 168) = 1;
    *(_QWORD *)(v4 + 184) = 0;
    *(_QWORD *)(v4 + 192) = 0;
    *(_DWORD *)(v4 + 200) = 8;
    v6 = (_QWORD *)malloc(8u);
    if ( !v6 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v6 = 0;
    }
    *v6 = 0;
    *(_QWORD *)(v4 + 184) = v6;
    *(_QWORD *)(v4 + 192) = 1;
    *(_QWORD *)(v4 + 208) = 0;
    *(_QWORD *)(v4 + 216) = 0;
    *(_DWORD *)(v4 + 224) = 8;
    v7 = (_QWORD *)malloc(8u);
    if ( !v7 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v7 = 0;
    }
    v8 = *(_BYTE **)a2;
    *(_QWORD *)(v4 + 208) = v7;
    v9 = (_BYTE *)(v4 + 256);
    *v7 = 0;
    v10 = *(_QWORD *)(a2 + 8);
    *(_QWORD *)v4 = off_49FB858;
    *(_QWORD *)(v4 + 232) = a1;
    *(_QWORD *)(v4 + 216) = 1;
    *(_QWORD *)(v4 + 240) = v4 + 256;
    if ( &v8[v10] && !v8 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v13[0] = v10;
    if ( v10 > 0xF )
    {
      v12 = sub_22409D0(v4 + 240, v13, 0);
      *(_QWORD *)(v4 + 240) = v12;
      v9 = (_BYTE *)v12;
      *(_QWORD *)(v4 + 256) = v13[0];
    }
    else
    {
      if ( v10 == 1 )
      {
        *(_BYTE *)(v4 + 256) = *v8;
LABEL_13:
        *(_QWORD *)(v4 + 248) = v10;
        v9[v10] = 0;
        return v4;
      }
      if ( !v10 )
        goto LABEL_13;
    }
    memcpy(v9, v8, v10);
    v10 = v13[0];
    v9 = *(_BYTE **)(v4 + 240);
    goto LABEL_13;
  }
  return v4;
}
