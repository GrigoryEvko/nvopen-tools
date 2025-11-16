// Function: sub_BDF4C0
// Address: 0xbdf4c0
//
unsigned __int64 __fastcall sub_BDF4C0(__int64 a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char *v5; // rsi
  unsigned __int8 **v6; // rdx
  const char *v7; // rcx
  int v8; // r8d
  unsigned __int8 *v9; // rdx
  int v10; // ecx
  const char *v11; // rsi
  unsigned __int8 *v12; // rax
  int v13; // edx
  unsigned __int8 *v14; // rax
  int v15; // edx
  unsigned __int64 result; // rax
  const char *v17; // rax
  int v18; // ecx
  __int64 v19; // r14
  _BYTE *v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r14
  _BYTE *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  _QWORD v26[4]; // [rsp+0h] [rbp-50h] BYREF
  char v27; // [rsp+20h] [rbp-30h]
  char v28; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int16)sub_AF18C0((__int64)a2) == 69 )
  {
    v4 = *(a2 - 16);
    v5 = a2 - 16;
    if ( (*(a2 - 16) & 2) != 0 )
    {
      v6 = (unsigned __int8 **)*((_QWORD *)a2 - 4);
      v7 = (const char *)v6;
      if ( !*v6 )
        goto LABEL_8;
      if ( !v6[2] )
      {
        v8 = **v6;
        if ( (unsigned int)(v8 - 25) <= 1 || (_BYTE)v8 == 7 )
        {
          v7 = (const char *)*((_QWORD *)a2 - 4);
          goto LABEL_8;
        }
LABEL_43:
        v28 = 1;
        v17 = "Count must be signed constant or DIVariable or DIExpression";
        goto LABEL_36;
      }
      goto LABEL_33;
    }
    v7 = &v5[-8 * ((v4 >> 2) & 0xF)];
    if ( *(_QWORD *)v7 )
    {
      if ( *((_QWORD *)v7 + 2) )
      {
LABEL_33:
        v28 = 1;
        v26[0] = "GenericSubrange can have any one of count or upperBound";
        v27 = 3;
        result = sub_BDD6D0((__int64 *)a1, (__int64)v26);
        if ( !*(_QWORD *)a1 )
          return result;
        return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
      }
      v18 = **(unsigned __int8 **)v7;
      if ( (unsigned int)(v18 - 25) > 1 && (_BYTE)v18 != 7 )
        goto LABEL_43;
      v7 = &v5[-8 * ((v4 >> 2) & 0xF)];
    }
LABEL_8:
    v9 = (unsigned __int8 *)*((_QWORD *)v7 + 1);
    if ( !v9 )
    {
      v28 = 1;
      v17 = "GenericSubrange must contain lowerBound";
      goto LABEL_36;
    }
    v10 = *v9;
    if ( (unsigned int)(v10 - 25) > 1 && (_BYTE)v10 != 7 )
    {
      v28 = 1;
      v17 = "LowerBound must be signed constant or DIVariable or DIExpression";
      goto LABEL_36;
    }
    if ( (*(a2 - 16) & 2) != 0 )
    {
      v11 = (const char *)*((_QWORD *)a2 - 4);
      v12 = (unsigned __int8 *)*((_QWORD *)v11 + 2);
      if ( !v12 )
        goto LABEL_15;
    }
    else
    {
      v11 = &v5[-8 * ((v4 >> 2) & 0xF)];
      v12 = (unsigned __int8 *)*((_QWORD *)v11 + 2);
      if ( !v12 )
        goto LABEL_15;
    }
    v13 = *v12;
    if ( (unsigned int)(v13 - 25) > 1 && (_BYTE)v13 != 7 )
    {
      v28 = 1;
      v17 = "UpperBound must be signed constant or DIVariable or DIExpression";
      goto LABEL_36;
    }
LABEL_15:
    v14 = (unsigned __int8 *)*((_QWORD *)v11 + 3);
    if ( v14 )
    {
      v15 = *v14;
      result = (unsigned int)(v15 - 25);
      if ( (unsigned int)result <= 1 || (_BYTE)v15 == 7 )
        return result;
      v28 = 1;
      v17 = "Stride must be signed constant or DIVariable or DIExpression";
    }
    else
    {
      v28 = 1;
      v17 = "GenericSubrange must contain stride";
    }
LABEL_36:
    v22 = *(_QWORD *)a1;
    v26[0] = v17;
    v27 = 3;
    if ( v22 )
    {
      sub_CA0E80(v26, v22);
      v23 = *(_BYTE **)(v22 + 32);
      if ( (unsigned __int64)v23 >= *(_QWORD *)(v22 + 24) )
      {
        sub_CB5D20(v22, 10);
      }
      else
      {
        *(_QWORD *)(v22 + 32) = v23 + 1;
        *v23 = 10;
      }
      v24 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= result;
      if ( v24 )
      {
        sub_A62C00(a2, v24, a1 + 16, *(_QWORD *)(a1 + 8));
        v25 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
          return sub_CB5D20(v25, 10);
        }
        else
        {
          *(_QWORD *)(v25 + 32) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
      return result;
    }
LABEL_32:
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 153) = 1;
    *(_BYTE *)(a1 + 152) |= result;
    return result;
  }
  v19 = *(_QWORD *)a1;
  v28 = 1;
  v26[0] = "invalid tag";
  v27 = 3;
  if ( !v19 )
    goto LABEL_32;
  sub_CA0E80(v26, v19);
  v20 = *(_BYTE **)(v19 + 32);
  if ( (unsigned __int64)v20 >= *(_QWORD *)(v19 + 24) )
  {
    sub_CB5D20(v19, 10);
  }
  else
  {
    *(_QWORD *)(v19 + 32) = v20 + 1;
    *v20 = 10;
  }
  v21 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= result;
  if ( v21 )
    return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
  return result;
}
