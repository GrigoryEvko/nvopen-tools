// Function: sub_1727650
// Address: 0x1727650
//
__int64 __fastcall sub_1727650(__int64 a1, __int64 a2, __int64 a3, int a4, double a5, double a6, double a7)
{
  __int64 v8; // rdx
  __int64 v9; // r14
  __int64 v10; // r15
  unsigned int v11; // r14d
  __int64 v12; // rax
  __int64 v14; // rax
  int v15; // eax
  int v16; // [rsp+Ch] [rbp-54h]
  int v17; // [rsp+10h] [rbp-50h]
  int v18; // [rsp+10h] [rbp-50h]
  __int64 v19; // [rsp+18h] [rbp-48h]
  bool v20; // [rsp+20h] [rbp-40h]
  bool v22; // [rsp+2Fh] [rbp-31h]

  v8 = 0;
  v9 = 0;
  v19 = a2;
  if ( *(_BYTE *)(a1 + 16) == 13 )
    v9 = a1;
  if ( *(_BYTE *)(a2 + 16) != 13 )
    a2 = 0;
  v22 = 0;
  if ( *(_BYTE *)(a3 + 16) == 13 )
    v8 = a3;
  v10 = v8;
  if ( v9 )
  {
    if ( *(_DWORD *)(v9 + 32) <= 0x40u )
    {
      v12 = *(_QWORD *)(v9 + 24);
      if ( v12 )
        v22 = (v12 & (v12 - 1)) == 0;
    }
    else
    {
      v17 = *(_DWORD *)(v9 + 32);
      if ( v17 != (unsigned int)sub_16A57B0(v9 + 24) )
        v22 = (unsigned int)sub_16A5940(v9 + 24) == 1;
    }
  }
  v20 = 0;
  if ( !a2 )
  {
LABEL_15:
    if ( v10 )
      goto LABEL_16;
LABEL_33:
    if ( a1 != a3 )
    {
      v11 = 0;
      if ( v19 != a3 )
        return v11;
LABEL_35:
      if ( a4 != 32 )
      {
        v11 |= 0x208u;
        goto LABEL_37;
      }
      v11 |= 0x104u;
LABEL_59:
      if ( !v20 )
        return v11;
      v15 = 544;
      goto LABEL_46;
    }
    goto LABEL_47;
  }
  if ( *(_DWORD *)(a2 + 32) > 0x40u )
  {
    v16 = *(_DWORD *)(a2 + 32);
    if ( v16 != (unsigned int)sub_16A57B0(a2 + 24) )
      v20 = (unsigned int)sub_16A5940(a2 + 24) == 1;
    goto LABEL_15;
  }
  v14 = *(_QWORD *)(a2 + 24);
  if ( !v14 )
    goto LABEL_15;
  v20 = (v14 & (v14 - 1)) == 0;
  if ( !v10 )
    goto LABEL_33;
LABEL_16:
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
  {
    v18 = *(_DWORD *)(v10 + 32);
    if ( v18 != (unsigned int)sub_16A57B0(v10 + 24) )
      goto LABEL_18;
LABEL_27:
    if ( a4 == 32 )
    {
      if ( v22 )
        return !v20 ? 466 : 986;
      else
        return !v20 ? 336 : 856;
    }
    else if ( v22 )
    {
      return !v20 ? 737 : 997;
    }
    else
    {
      return !v20 ? 672 : 932;
    }
  }
  if ( !*(_QWORD *)(v10 + 24) )
    goto LABEL_27;
LABEL_18:
  if ( a1 == a3 )
  {
LABEL_47:
    if ( a4 == 32 )
    {
      if ( !v22 )
      {
        v11 = 65;
        if ( v19 != a3 )
          goto LABEL_41;
        v11 = 325;
        goto LABEL_59;
      }
      v11 = 225;
    }
    else
    {
      if ( !v22 )
      {
        v11 = 130;
        if ( v19 != a3 )
          goto LABEL_41;
        v11 = 650;
LABEL_37:
        v15 = 272;
        if ( !v20 )
          return v11;
        goto LABEL_46;
      }
      v11 = 210;
    }
    goto LABEL_40;
  }
  if ( !v9 || v10 != sub_15A2CF0((__int64 *)v9, v10, a5, a6, a7) )
  {
    v11 = 0;
LABEL_40:
    if ( v19 != a3 )
      goto LABEL_41;
    goto LABEL_35;
  }
  if ( a4 != 32 )
  {
    v11 = 128;
    if ( v19 == a3 )
    {
      v11 = 648;
      goto LABEL_37;
    }
    goto LABEL_41;
  }
  v11 = 64;
  if ( v19 == a3 )
  {
    v11 = 324;
    goto LABEL_59;
  }
LABEL_41:
  if ( a2 && v10 && v10 == sub_15A2CF0((__int64 *)a2, v10, a5, a6, a7) )
  {
    v15 = 256;
    if ( a4 != 32 )
      v15 = 512;
LABEL_46:
    v11 |= v15;
  }
  return v11;
}
