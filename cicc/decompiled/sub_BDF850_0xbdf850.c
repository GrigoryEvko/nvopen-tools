// Function: sub_BDF850
// Address: 0xbdf850
//
unsigned __int64 __fastcall sub_BDF850(__int64 *a1, const char *a2)
{
  unsigned __int8 v4; // cl
  const char *v5; // rdi
  const char *v6; // rax
  unsigned __int8 *v7; // rsi
  unsigned __int64 v8; // rsi
  __int64 v9; // r8
  unsigned __int64 result; // rax
  unsigned __int8 *v11; // rsi
  unsigned __int64 v12; // rax
  const char *v13; // rax
  __int64 v14; // rsi
  unsigned __int8 *v15; // rsi
  unsigned __int64 v16; // rax
  __int64 v17; // rsi
  unsigned __int8 *v18; // rsi
  unsigned __int64 v19; // rax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // r14
  _BYTE *v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rdi
  _QWORD v26[4]; // [rsp+0h] [rbp-50h] BYREF
  char v27; // [rsp+20h] [rbp-30h]
  char v28; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int16)sub_AF18C0((__int64)a2) != 33 )
  {
    v22 = *a1;
    v28 = 1;
    v26[0] = "invalid tag";
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
      v24 = *a1;
      result = *((unsigned __int8 *)a1 + 154);
      *((_BYTE *)a1 + 153) = 1;
      *((_BYTE *)a1 + 152) |= result;
      if ( v24 )
      {
        sub_A62C00(a2, v24, (__int64)(a1 + 2), a1[1]);
        v25 = *a1;
        result = *(_QWORD *)(*a1 + 32);
        if ( result >= *(_QWORD *)(*a1 + 24) )
        {
          return sub_CB5D20(v25, 10);
        }
        else
        {
          *(_QWORD *)(v25 + 32) = result + 1;
          *(_BYTE *)result = 10;
        }
      }
    }
    else
    {
      result = *((unsigned __int8 *)a1 + 154);
      *((_BYTE *)a1 + 153) = 1;
      *((_BYTE *)a1 + 152) |= result;
    }
    return result;
  }
  v4 = *(a2 - 16);
  v5 = a2 - 16;
  if ( (v4 & 2) != 0 )
  {
    v6 = (const char *)*((_QWORD *)a2 - 4);
    v7 = (unsigned __int8 *)*((_QWORD *)v6 + 3);
    if ( !v7 )
      goto LABEL_9;
  }
  else
  {
    v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
    v7 = (unsigned __int8 *)*((_QWORD *)v6 + 3);
    if ( !v7 )
      goto LABEL_9;
  }
  v8 = *v7;
  if ( (unsigned __int8)v8 > 0x24u || (v9 = 0x140000F000LL, !_bittest64(&v9, v8)) )
  {
    v28 = 1;
    v26[0] = "BaseType must be a type";
    v27 = 3;
    return sub_BDD6D0(a1, (__int64)v26);
  }
LABEL_9:
  v11 = (unsigned __int8 *)*((_QWORD *)v6 + 4);
  if ( v11 )
  {
    v12 = *v11;
    if ( (unsigned __int8)v12 > 0x1Au || (v14 = 100663426, !_bittest64(&v14, v12)) )
    {
      v28 = 1;
      v13 = "LowerBound must be signed constant or DIVariable or DIExpression";
      goto LABEL_12;
    }
    if ( (v4 & 2) != 0 )
      v6 = (const char *)*((_QWORD *)a2 - 4);
    else
      v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
  }
  v15 = (unsigned __int8 *)*((_QWORD *)v6 + 5);
  if ( v15 )
  {
    v16 = *v15;
    if ( (unsigned __int8)v16 > 0x1Au || (v17 = 100663426, !_bittest64(&v17, v16)) )
    {
      v28 = 1;
      v13 = "UpperBound must be signed constant or DIVariable or DIExpression";
      goto LABEL_12;
    }
    if ( (v4 & 2) != 0 )
      v6 = (const char *)*((_QWORD *)a2 - 4);
    else
      v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
  }
  v18 = (unsigned __int8 *)*((_QWORD *)v6 + 6);
  if ( !v18 )
    goto LABEL_27;
  v19 = *v18;
  if ( (unsigned __int8)v19 > 0x1Au || (v20 = 100663426, !_bittest64(&v20, v19)) )
  {
    v28 = 1;
    v13 = "Stride must be signed constant or DIVariable or DIExpression";
LABEL_12:
    v26[0] = v13;
    v27 = 3;
    result = sub_BDD6D0(a1, (__int64)v26);
    if ( *a1 )
      return (unsigned __int64)sub_BD9900(a1, a2);
    return result;
  }
  if ( (v4 & 2) != 0 )
    v6 = (const char *)*((_QWORD *)a2 - 4);
  else
    v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
LABEL_27:
  result = *((_QWORD *)v6 + 7);
  if ( result )
  {
    result = *(unsigned __int8 *)result;
    if ( (unsigned __int8)result > 0x1Au || (v21 = 100663426, !_bittest64(&v21, result)) )
    {
      v28 = 1;
      v13 = "Bias must be signed constant or DIVariable or DIExpression";
      goto LABEL_12;
    }
  }
  return result;
}
