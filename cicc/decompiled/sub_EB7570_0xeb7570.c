// Function: sub_EB7570
// Address: 0xeb7570
//
__int64 __fastcall sub_EB7570(__int64 a1, unsigned int a2)
{
  __int64 v3; // r12
  __int64 v4; // rdi
  _DWORD *v5; // rax
  unsigned int v6; // r13d
  char v7; // al
  __int64 *v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v13; // r13
  __int64 v14; // rdx
  int v15; // eax
  const char *v16; // rax
  unsigned __int64 v17; // rax
  __int64 v18; // rdi
  void *v19; // rax
  __int64 v20; // rcx
  __int64 v21; // [rsp+0h] [rbp-90h]
  __int64 v22; // [rsp+8h] [rbp-88h]
  __int64 v23; // [rsp+10h] [rbp-80h] BYREF
  unsigned __int64 v24; // [rsp+18h] [rbp-78h] BYREF
  const char *v25; // [rsp+20h] [rbp-70h] BYREF
  const char *v26; // [rsp+28h] [rbp-68h]
  const char *v27[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v28; // [rsp+50h] [rbp-40h]

  if ( !*(_BYTE *)(a1 + 869) && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  v25 = 0;
  v26 = 0;
  v3 = sub_ECD690(a1 + 40);
  if ( (unsigned __int8)sub_EB61F0(a1, (__int64 *)&v25) )
  {
    v27[0] = "expected identifier in directive";
    v28 = 259;
    return (unsigned int)sub_ECE0E0(a1, v27, 0, 0);
  }
  v4 = *(_QWORD *)(a1 + 224);
  v28 = 261;
  v27[0] = v25;
  v27[1] = v26;
  v22 = sub_E6C460(v4, v27);
  v27[0] = "expected comma";
  v28 = 259;
  if ( (unsigned __int8)sub_ECE210(a1, 26, v27) )
    return 1;
  v21 = sub_ECD690(a1 + 40);
  if ( (unsigned __int8)sub_EAC8B0(a1, &v23) )
    return 1;
  v5 = *(_DWORD **)(a1 + 48);
  v24 = 0;
  if ( *v5 == 26 )
  {
    sub_EABFE0(a1);
    v13 = sub_ECD690(a1 + 40);
    if ( (unsigned __int8)sub_EAC8B0(a1, &v24) )
      return 1;
    v14 = *(_QWORD *)(a1 + 184);
    v15 = *(_DWORD *)(v14 + 284);
    if ( v15 )
    {
      if ( (_BYTE)a2 )
      {
        if ( v15 != 1 )
          goto LABEL_7;
        goto LABEL_28;
      }
    }
    else if ( (_BYTE)a2 )
    {
      HIBYTE(v28) = 1;
      v16 = "alignment not supported on this target";
      goto LABEL_24;
    }
    if ( !*(_BYTE *)(v14 + 281) )
      goto LABEL_7;
LABEL_28:
    if ( v24 && (v24 & (v24 - 1)) == 0 )
    {
      _BitScanReverse64(&v17, v24);
      v24 = (int)(63 - (v17 ^ 0x3F));
      goto LABEL_7;
    }
    HIBYTE(v28) = 1;
    v16 = "alignment must be a power of 2";
LABEL_24:
    v27[0] = v16;
    LOBYTE(v28) = 3;
    return (unsigned int)sub_ECDA70(a1, v13, v27, 0, 0);
  }
LABEL_7:
  v6 = sub_ECE000(a1);
  if ( (_BYTE)v6 )
    return 1;
  if ( v23 < 0 )
  {
    v27[0] = "size must be non-negative";
    v28 = 259;
    return (unsigned int)sub_ECDA70(a1, v21, v27, 0, 0);
  }
  else
  {
    v7 = *(_BYTE *)(v22 + 8);
    if ( (v7 & 4) != 0 )
    {
      if ( (*(_BYTE *)(v22 + 9) & 0x70) == 0x20 )
      {
        *(_WORD *)(v22 + 8) &= 0x8FFBu;
        *(_QWORD *)(v22 + 24) = 0;
        *(_QWORD *)v22 = 0;
      }
      else
      {
        *(_QWORD *)v22 = 0;
        *(_BYTE *)(v22 + 8) = v7 & 0xFB;
      }
    }
    else if ( *(_QWORD *)v22
           || (*(_BYTE *)(v22 + 9) & 0x70) == 0x20
           && *(char *)(v22 + 8) >= 0
           && (v18 = *(_QWORD *)(v22 + 24),
               *(_BYTE *)(v22 + 8) = v7 | 8,
               v19 = sub_E807D0(v18),
               (*(_QWORD *)v22 = v19) != 0) )
    {
      v27[0] = "invalid symbol redefinition";
      v28 = 259;
      return (unsigned int)sub_ECDA70(a1, v3, v27, 0, 0);
    }
    v8 = *(__int64 **)(a1 + 232);
    v9 = *v8;
    v10 = 1LL << v24;
    if ( (_BYTE)a2 )
    {
      v20 = 0xFFFFFFFFLL;
      if ( v10 )
      {
        _BitScanReverse64(&v10, v10);
        v20 = 63 - ((unsigned int)v10 ^ 0x3F);
      }
      (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64))(v9 + 488))(v8, v22, v23, v20);
    }
    else
    {
      v11 = 0xFFFFFFFFLL;
      if ( v10 )
      {
        _BitScanReverse64(&v10, v10);
        v11 = 63 - ((unsigned int)v10 ^ 0x3F);
      }
      v6 = a2;
      (*(void (__fastcall **)(__int64 *, __int64, __int64, __int64))(v9 + 480))(v8, v22, v23, v11);
    }
  }
  return v6;
}
