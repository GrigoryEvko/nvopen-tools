// Function: sub_8960C0
// Address: 0x8960c0
//
void __fastcall sub_8960C0(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // r12
  char v6; // al
  __int64 v7; // rsi
  int v8; // edx
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 *v12; // r12
  _QWORD *i; // r14
  __int64 v14; // rdx
  char v15; // al
  __int64 *v16; // rdi
  _QWORD *v17; // rdx
  __int64 v18; // rsi
  char v19; // al
  int v20; // edx
  __int64 v21; // rax
  _QWORD *v22; // [rsp-50h] [rbp-50h]
  __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 *v24; // [rsp-40h] [rbp-40h]

  if ( *(_QWORD *)(a1 + 96) )
    return;
  v3 = *(_QWORD *)(a2 + 96);
  if ( (unsigned __int8)sub_877F80(a1) == 1 )
  {
    v5 = *(_QWORD *)(v3 + 8);
    if ( !v5 )
      goto LABEL_26;
  }
  else
  {
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 88) + 174LL) != 3 )
    {
      v4 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 88) + 168LL) + 152LL);
      if ( !v4 || (*(_BYTE *)(v4 + 29) & 0x20) != 0 || (v5 = sub_883800(v3 + 192, *(_QWORD *)a1)) == 0 )
      {
LABEL_26:
        if ( (*(_BYTE *)(a1 + 84) & 0x10) == 0 )
          return;
        goto LABEL_27;
      }
      do
      {
        v6 = *(_BYTE *)(v5 + 80);
        if ( v6 == 10 || v6 == 17 )
        {
          v7 = v5;
          goto LABEL_14;
        }
        v5 = *(_QWORD *)(v5 + 32);
      }
      while ( v5 );
      if ( (*(_BYTE *)(a1 + 84) & 0x10) != 0 )
      {
LABEL_27:
        ++unk_4F07488;
        return;
      }
      goto LABEL_24;
    }
    v16 = *(__int64 **)(a2 + 96);
    v17 = (_QWORD *)v16[5];
    if ( v17 )
    {
      while ( 1 )
      {
        v5 = v17[1];
        v6 = *(_BYTE *)(v5 + 80);
        if ( v6 != 16 && *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(v5 + 96) + 56LL) + 64LL) == dword_4F06650[0] )
          break;
        v17 = (_QWORD *)*v17;
        if ( !v17 )
          goto LABEL_50;
      }
      v7 = 0;
      goto LABEL_14;
    }
LABEL_50:
    v18 = *v16;
    if ( !*v16 )
      goto LABEL_26;
    while ( 1 )
    {
      v19 = *(_BYTE *)(v18 + 80);
      if ( v19 == 10 )
        break;
      if ( v19 == 17 )
      {
        v5 = *(_QWORD *)(v18 + 88);
        if ( v5 )
        {
          v20 = 1;
          goto LABEL_56;
        }
      }
LABEL_53:
      v18 = *(_QWORD *)(v18 + 16);
      if ( !v18 )
        goto LABEL_26;
    }
    v5 = v18;
    v20 = 0;
LABEL_56:
    while ( 1 )
    {
      v21 = *(_QWORD *)(v5 + 96);
      if ( v21 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(v21 + 56) + 64LL) == dword_4F06650[0] )
          break;
      }
      if ( v20 )
      {
        v5 = *(_QWORD *)(v5 + 8);
        if ( v5 )
          continue;
      }
      goto LABEL_53;
    }
  }
  v6 = *(_BYTE *)(v5 + 80);
  v7 = 0;
LABEL_14:
  v8 = 0;
  if ( v6 != 17 )
    goto LABEL_18;
  v5 = *(_QWORD *)(v5 + 88);
  if ( !v5 )
  {
LABEL_21:
    if ( (*(_BYTE *)(a1 + 84) & 0x10) != 0 )
      goto LABEL_27;
    if ( !v7 )
      return;
    v6 = *(_BYTE *)(v7 + 80);
LABEL_24:
    if ( (unsigned __int8)(v6 - 8) <= 1u )
      sub_6851C0(0xA8u, dword_4F07508);
    return;
  }
  v6 = *(_BYTE *)(v5 + 80);
  v8 = 1;
LABEL_18:
  while ( 1 )
  {
    if ( v6 == 10 )
    {
      v9 = *(_QWORD *)(v5 + 96);
      if ( v9 )
      {
        if ( *(_DWORD *)(*(_QWORD *)(v9 + 56) + 64LL) == dword_4F06650[0] )
          break;
      }
    }
    if ( v8 )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( !v5 )
        goto LABEL_21;
    }
    else
    {
      v5 = *(_QWORD *)(v5 + 32);
      if ( !v5 )
        goto LABEL_21;
    }
    v6 = *(_BYTE *)(v5 + 80);
  }
  v10 = *(_QWORD *)(a1 + 88);
  v23 = *(_QWORD *)(v5 + 88);
  v11 = sub_880C60();
  *(_QWORD *)(v11 + 32) = v5;
  v22 = (_QWORD *)v11;
  v24 = *(__int64 **)(*(_QWORD *)(v5 + 96) + 56LL);
  v12 = v24;
  while ( 1 )
  {
    for ( i = (_QWORD *)v12[9]; i; i = (_QWORD *)*i )
      sub_5EDDD0(v10, i[1]);
    v14 = v12[11];
    if ( !v14 )
      break;
    switch ( *(_BYTE *)(v14 + 80) )
    {
      case 4:
      case 5:
        v12 = *(__int64 **)(*(_QWORD *)(v14 + 96) + 80LL);
        break;
      case 6:
        v12 = *(__int64 **)(*(_QWORD *)(v14 + 96) + 32LL);
        break;
      case 9:
      case 0xA:
        v12 = *(__int64 **)(*(_QWORD *)(v14 + 96) + 56LL);
        break;
      case 0x13:
      case 0x14:
      case 0x15:
      case 0x16:
        v12 = *(__int64 **)(v14 + 88);
        break;
      default:
        v12 = 0;
        break;
    }
  }
  *v22 = v24[21];
  v24[21] = (__int64)v22;
  v22[3] = a1;
  *(_QWORD *)(a1 + 96) = v22;
  *(_BYTE *)(v10 + 195) |= 1u;
  *(_QWORD *)(v10 + 248) = *(_QWORD *)(v23 + 248);
  *(_QWORD *)(v10 + 216) = *(_QWORD *)(v23 + 216);
  v15 = *(_BYTE *)(v23 + 206);
  if ( (v15 & 0x10) != 0 )
  {
    *(_BYTE *)(v10 + 206) |= 0x10u;
    v15 = *(_BYTE *)(v23 + 206);
  }
  if ( (v15 & 8) != 0 )
    *(_BYTE *)(v10 + 206) |= 8u;
  sub_88FC40(*(_QWORD *)(v24[22] + 152), v10, v24);
}
