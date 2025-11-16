// Function: sub_2B16650
// Address: 0x2b16650
//
unsigned __int8 **__fastcall sub_2B16650(unsigned __int8 **a1, __int64 a2, unsigned int a3)
{
  unsigned __int8 **v3; // r8
  unsigned int v4; // r15d
  unsigned int v5; // r14d
  char v7; // bl
  __int64 v8; // r10
  __int64 v9; // rax
  char v10; // r9
  unsigned __int8 **v11; // rax
  unsigned __int8 **v12; // rcx
  unsigned __int8 **v13; // r10
  char v14; // r11
  unsigned __int8 **v15; // rsi
  int v16; // edx
  int v18; // edx
  int v19; // edx
  int v20; // edx
  char v21; // di
  int v22; // eax
  int v23; // eax
  int v24; // eax
  char v25; // [rsp+1h] [rbp-29h]

  v3 = a1;
  v4 = HIBYTE(a3);
  v5 = HIWORD(a3);
  v7 = a3;
  v25 = BYTE1(a3);
  v8 = (a2 - (__int64)a1) >> 5;
  v9 = (a2 - (__int64)a1) >> 3;
  if ( v8 > 0 )
  {
    v10 = a3;
    v11 = a1 + 2;
    v12 = a1 + 1;
    v13 = &a1[4 * v8];
    v14 = BYTE1(a3) & BYTE2(a3);
    v15 = a1 + 3;
    while ( 1 )
    {
      v16 = **(v11 - 2);
      if ( (_BYTE)v16 != 13 )
        break;
      v18 = **(v11 - 1);
      if ( (_BYTE)v18 != 13 )
      {
        if ( !v10 )
          return v12;
        v21 = v14;
LABEL_20:
        if ( !v21 && !(_BYTE)v4 || (unsigned int)(v18 - 12) > 1 )
          return v12;
        v19 = **v11;
        if ( (_BYTE)v19 != 13 )
          goto LABEL_24;
        goto LABEL_13;
      }
      v19 = **v11;
      if ( (_BYTE)v19 != 13 )
      {
        if ( !v10 )
          return v11;
LABEL_38:
        v21 = v14;
LABEL_24:
        if ( !v21 && !(_BYTE)v4 || (unsigned int)(v19 - 12) > 1 )
          return v11;
        v20 = *v11[1];
        if ( (_BYTE)v20 != 13 )
          goto LABEL_28;
        goto LABEL_31;
      }
LABEL_13:
      v20 = *v11[1];
      if ( (_BYTE)v20 != 13 )
      {
        if ( !v10 )
          return v15;
        v21 = v14;
LABEL_28:
        if ( !v21 && !(_BYTE)v4 || (unsigned int)(v20 - 12) > 1 )
          return v15;
      }
LABEL_31:
      v3 += 4;
      v11 += 4;
      v12 += 4;
      v15 += 4;
      if ( v3 == v13 )
      {
        v9 = (a2 - (__int64)v3) >> 3;
        goto LABEL_33;
      }
    }
    if ( !v10 )
      return v3;
    v21 = v14;
    if ( !v14 && !(_BYTE)v4 )
      return v3;
    if ( (unsigned int)(v16 - 12) > 1 )
      return v3;
    v18 = **(v11 - 1);
    if ( (_BYTE)v18 == 13 )
    {
      v19 = **v11;
      if ( (_BYTE)v19 != 13 )
        goto LABEL_38;
      goto LABEL_13;
    }
    goto LABEL_20;
  }
LABEL_33:
  if ( v9 != 2 )
  {
    if ( v9 != 3 )
    {
      if ( v9 != 1 )
        return (unsigned __int8 **)a2;
      goto LABEL_52;
    }
    v22 = **v3;
    if ( (_BYTE)v22 != 13 && (!v7 || (!v25 || !(_BYTE)v5) && !(_BYTE)v4 || (unsigned int)(v22 - 12) > 1) )
      return v3;
    ++v3;
  }
  v23 = **v3;
  if ( (_BYTE)v23 != 13 && (!v7 || (!v25 || !(_BYTE)v5) && !(_BYTE)v4 || (unsigned int)(v23 - 12) > 1) )
    return v3;
  ++v3;
LABEL_52:
  v24 = **v3;
  if ( (_BYTE)v24 != 13 )
  {
    if ( v7 && (v25 && (_BYTE)v5 || (_BYTE)v4) && (unsigned int)(v24 - 12) <= 1 )
      return (unsigned __int8 **)a2;
    return v3;
  }
  return (unsigned __int8 **)a2;
}
