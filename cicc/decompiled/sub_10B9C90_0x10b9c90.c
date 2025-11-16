// Function: sub_10B9C90
// Address: 0x10b9c90
//
__int64 __fastcall sub_10B9C90(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // r9
  __int64 v6; // r15
  __int64 v7; // r14
  bool v8; // cl
  bool v9; // r10
  unsigned int v10; // edx
  int v11; // eax
  char v12; // al
  unsigned int v13; // r12d
  int v14; // eax
  int v16; // eax
  int v17; // eax
  __int64 v18; // rdx
  _BYTE *v19; // rax
  __int64 v20; // rdx
  _BYTE *v21; // rax
  __int64 v22; // rdx
  _BYTE *v23; // rax
  __int64 v24; // [rsp+0h] [rbp-50h]
  bool v25; // [rsp+8h] [rbp-48h]
  __int64 v26; // [rsp+8h] [rbp-48h]
  bool v27; // [rsp+10h] [rbp-40h]
  bool v28; // [rsp+10h] [rbp-40h]
  bool v29; // [rsp+10h] [rbp-40h]
  __int64 v30; // [rsp+10h] [rbp-40h]
  __int64 v31; // [rsp+10h] [rbp-40h]
  __int64 v32; // [rsp+10h] [rbp-40h]
  unsigned int v33; // [rsp+18h] [rbp-38h]

  v4 = a1 + 24;
  if ( *(_BYTE *)a1 != 17 )
  {
    v22 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL) - 17;
    if ( (unsigned int)v22 <= 1 && *(_BYTE *)a1 <= 0x15u && (v23 = sub_AD7630(a1, 0, v22)) != 0 && *v23 == 17 )
      v4 = (__int64)(v23 + 24);
    else
      v4 = 0;
  }
  v6 = a2 + 24;
  if ( *(_BYTE *)a2 != 17 )
  {
    v20 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a2 + 8) + 8LL) - 17;
    if ( (unsigned int)v20 <= 1
      && *(_BYTE *)a2 <= 0x15u
      && (v32 = v4, v21 = sub_AD7630(a2, 0, v20), v4 = v32, v21)
      && *v21 == 17 )
    {
      v6 = (__int64)(v21 + 24);
    }
    else
    {
      v6 = 0;
    }
  }
  v7 = a3 + 24;
  if ( *(_BYTE *)a3 != 17 )
  {
    v18 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(a3 + 8) + 8LL) - 17;
    if ( (unsigned int)v18 <= 1
      && *(_BYTE *)a3 <= 0x15u
      && (v31 = v4, v19 = sub_AD7630(a3, 0, v18), v4 = v31, v19)
      && *v19 == 17 )
    {
      v7 = (__int64)(v19 + 24);
    }
    else
    {
      v7 = 0;
    }
  }
  v8 = 0;
  if ( v4 )
  {
    if ( *(_DWORD *)(v4 + 8) > 0x40u )
    {
      v30 = v4;
      v17 = sub_C44630(v4);
      v4 = v30;
      v8 = v17 == 1;
    }
    else if ( *(_QWORD *)v4 )
    {
      v8 = (*(_QWORD *)v4 & (*(_QWORD *)v4 - 1LL)) == 0;
    }
  }
  v9 = 0;
  if ( !v6 )
  {
LABEL_12:
    if ( v7 )
      goto LABEL_13;
LABEL_30:
    if ( a1 != a3 )
    {
      v13 = 0;
      if ( a2 != a3 )
        return v13;
LABEL_60:
      if ( a4 != 32 )
      {
        v13 |= 0x208u;
        goto LABEL_62;
      }
      v13 |= 0x104u;
LABEL_77:
      v14 = 544;
      if ( !v9 )
        return v13;
      goto LABEL_27;
    }
    goto LABEL_56;
  }
  if ( *(_DWORD *)(v6 + 8) <= 0x40u )
  {
    if ( *(_QWORD *)v6 )
      v9 = (*(_QWORD *)v6 & (*(_QWORD *)v6 - 1LL)) == 0;
    goto LABEL_12;
  }
  v26 = v4;
  v29 = v8;
  v16 = sub_C44630(v6);
  v4 = v26;
  v8 = v29;
  v9 = v16 == 1;
  if ( !v7 )
    goto LABEL_30;
LABEL_13:
  v10 = *(_DWORD *)(v7 + 8);
  if ( v10 > 0x40 )
  {
    v33 = *(_DWORD *)(v7 + 8);
    v24 = v4;
    v25 = v9;
    v27 = v8;
    v11 = sub_C444A0(v7);
    v10 = v33;
    v8 = v27;
    v9 = v25;
    v4 = v24;
    if ( v33 != v11 )
      goto LABEL_15;
LABEL_35:
    if ( a4 == 32 )
    {
      if ( v8 )
        return !v9 ? 466 : 986;
      else
        return !v9 ? 336 : 856;
    }
    else if ( v8 )
    {
      return !v9 ? 737 : 997;
    }
    else
    {
      return !v9 ? 672 : 932;
    }
  }
  if ( !*(_QWORD *)v7 )
    goto LABEL_35;
LABEL_15:
  if ( a1 == a3 )
  {
LABEL_56:
    if ( a4 == 32 )
    {
      v13 = 225;
      if ( !v8 )
      {
        v13 = 65;
        if ( a2 != a3 )
          goto LABEL_21;
        v13 = 325;
        goto LABEL_77;
      }
    }
    else
    {
      if ( !v8 )
      {
        v13 = 130;
        if ( a2 != a3 )
          goto LABEL_21;
        v13 = 650;
LABEL_62:
        v14 = 272;
        if ( !v9 )
          return v13;
        goto LABEL_27;
      }
      v13 = 210;
    }
    goto LABEL_59;
  }
  if ( !v4 )
    goto LABEL_65;
  if ( v10 > 0x40 )
  {
    v28 = v9;
    v12 = sub_C446F0((__int64 *)v7, (__int64 *)v4);
    v9 = v28;
    if ( v12 )
      goto LABEL_19;
    goto LABEL_65;
  }
  if ( (*(_QWORD *)v7 & ~*(_QWORD *)v4) != 0 )
  {
LABEL_65:
    v13 = 0;
LABEL_59:
    if ( a2 != a3 )
      goto LABEL_21;
    goto LABEL_60;
  }
LABEL_19:
  if ( a4 == 32 )
  {
    v13 = 64;
    if ( a2 != a3 )
      goto LABEL_21;
    v13 = 324;
    goto LABEL_77;
  }
  v13 = 128;
  if ( a2 == a3 )
  {
    v13 = 648;
    goto LABEL_62;
  }
LABEL_21:
  if ( v6 && v7 )
  {
    if ( *(_DWORD *)(v7 + 8) <= 0x40u )
    {
      if ( (*(_QWORD *)v7 & ~*(_QWORD *)v6) != 0 )
        return v13;
      goto LABEL_25;
    }
    if ( (unsigned __int8)sub_C446F0((__int64 *)v7, (__int64 *)v6) )
    {
LABEL_25:
      v14 = 256;
      if ( a4 != 32 )
        v14 = 512;
LABEL_27:
      v13 |= v14;
    }
  }
  return v13;
}
