// Function: sub_2DB5330
// Address: 0x2db5330
//
_BYTE *__fastcall sub_2DB5330(
        _BYTE *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        _QWORD *a9)
{
  signed __int64 v9; // rax
  _BYTE *v10; // r12
  __int64 v11; // rsi
  __int64 v13; // rax
  _BYTE *v14; // r15
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rsi
  __int64 v24; // rax
  __int64 v25; // rsi
  __int64 v26; // rax
  _BYTE *v28; // [rsp+8h] [rbp-38h]

  v9 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)a1) >> 3);
  v10 = a1;
  if ( v9 >> 2 <= 0 )
  {
LABEL_26:
    if ( v9 != 2 )
    {
      if ( v9 != 3 )
      {
        if ( v9 != 1 )
          return (_BYTE *)a2;
        goto LABEL_43;
      }
      if ( *v10 != 1 )
      {
        if ( *(_BYTE *)a7 )
          return v10;
        if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
          return v10;
        v21 = *(unsigned int *)(a7 + 8);
        if ( (unsigned int)(v21 - 1) <= 0x3FFFFFFE )
          return v10;
        v22 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v21);
        if ( !(unsigned __int8)sub_2EA6A70(*a9, v22, 0) )
          return v10;
      }
      v10 += 40;
    }
    if ( *v10 != 1 )
    {
      if ( *(_BYTE *)a7 )
        return v10;
      if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
        return v10;
      v23 = *(unsigned int *)(a7 + 8);
      if ( (unsigned int)(v23 - 1) <= 0x3FFFFFFE )
        return v10;
      v24 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v23);
      if ( !(unsigned __int8)sub_2EA6A70(*a9, v24, 0) )
        return v10;
    }
    v10 += 40;
LABEL_43:
    if ( *v10 != 1 )
    {
      if ( !*(_BYTE *)a7 && (*(_BYTE *)(a7 + 3) & 0x10) == 0 )
      {
        v25 = *(unsigned int *)(a7 + 8);
        if ( (unsigned int)(v25 - 1) > 0x3FFFFFFE )
        {
          v26 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v25);
          if ( (unsigned __int8)sub_2EA6A70(*a9, v26, 0) )
            return (_BYTE *)a2;
        }
      }
      return v10;
    }
    return (_BYTE *)a2;
  }
  v28 = &a1[160 * (v9 >> 2)];
  while ( 1 )
  {
    if ( *v10 != 1 )
    {
      if ( *(_BYTE *)a7 )
        return v10;
      if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
        return v10;
      v11 = *(unsigned int *)(a7 + 8);
      if ( (unsigned int)(v11 - 1) <= 0x3FFFFFFE )
        return v10;
      v13 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v11);
      if ( !(unsigned __int8)sub_2EA6A70(*a9, v13, 0) )
        return v10;
    }
    v14 = v10 + 40;
    if ( v10[40] != 1 )
    {
      if ( *(_BYTE *)a7 )
        return v14;
      if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
        return v14;
      v15 = *(unsigned int *)(a7 + 8);
      if ( (unsigned int)(v15 - 1) <= 0x3FFFFFFE )
        return v14;
      v16 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v15);
      if ( !(unsigned __int8)sub_2EA6A70(*a9, v16, 0) )
        return v14;
    }
    v14 = v10 + 80;
    if ( v10[80] != 1 )
    {
      if ( *(_BYTE *)a7 )
        return v14;
      if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
        return v14;
      v17 = *(unsigned int *)(a7 + 8);
      if ( (unsigned int)(v17 - 1) <= 0x3FFFFFFE )
        return v14;
      v18 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v17);
      if ( !(unsigned __int8)sub_2EA6A70(*a9, v18, 0) )
        return v14;
    }
    v14 = v10 + 120;
    if ( v10[120] != 1 )
    {
      if ( *(_BYTE *)a7 )
        return v14;
      if ( (*(_BYTE *)(a7 + 3) & 0x10) != 0 )
        return v14;
      v19 = *(unsigned int *)(a7 + 8);
      if ( (unsigned int)(v19 - 1) <= 0x3FFFFFFE )
        return v14;
      v20 = sub_2EBEE10(*(_QWORD *)(a8 + 96), v19);
      if ( !(unsigned __int8)sub_2EA6A70(*a9, v20, 0) )
        return v14;
    }
    v10 += 160;
    if ( v10 == v28 )
    {
      v9 = 0xCCCCCCCCCCCCCCCDLL * ((a2 - (__int64)v10) >> 3);
      goto LABEL_26;
    }
  }
}
