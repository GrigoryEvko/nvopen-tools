// Function: sub_BDF0A0
// Address: 0xbdf0a0
//
unsigned __int64 __fastcall sub_BDF0A0(_BYTE *a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char *v5; // rbx
  const char *v6; // rdx
  unsigned __int8 *v7; // rax
  unsigned __int64 v8; // rax
  const char *v9; // rax
  __int64 v10; // r14
  _BYTE *v11; // rax
  __int64 v12; // rsi
  unsigned __int64 result; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rax
  unsigned int v18; // edx
  __int64 v19; // rax
  unsigned __int8 v20; // al
  __int64 v21; // rsi
  unsigned __int8 *v22; // rdi
  const char *v23; // rcx
  unsigned __int8 v24; // di
  unsigned __int8 *v25; // rsi
  unsigned __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // r14
  _BYTE *v30; // rax
  __int64 v31; // rdx
  unsigned __int8 *v32; // rsi
  unsigned __int8 v33; // si
  unsigned __int8 *v34; // rsi
  __int64 v35; // rdx
  _QWORD v36[4]; // [rsp+0h] [rbp-50h] BYREF
  char v37; // [rsp+20h] [rbp-30h]
  char v38; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int16)sub_AF18C0((__int64)a2) == 33 )
  {
    v4 = *(a2 - 16);
    v5 = a2 - 16;
    if ( (v4 & 2) != 0 )
    {
      v6 = (const char *)*((_QWORD *)a2 - 4);
      v7 = *(unsigned __int8 **)v6;
      if ( !*(_QWORD *)v6 )
        goto LABEL_15;
    }
    else
    {
      v6 = &v5[-8 * ((v4 >> 2) & 0xF)];
      v7 = *(unsigned __int8 **)v6;
      if ( !*(_QWORD *)v6 )
        goto LABEL_15;
    }
    if ( !*((_QWORD *)v6 + 2) )
    {
      v8 = *v7;
      if ( (unsigned __int8)v8 > 0x1Au || (v35 = 100663426, !_bittest64(&v35, v8)) )
      {
        v38 = 1;
        v9 = "Count must be signed constant or DIVariable or DIExpression";
        goto LABEL_7;
      }
LABEL_15:
      v15 = sub_AF2780((__int64)a2);
      v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v15 & 0xFFFFFFFFFFFFFFF8LL) != 0 && (v15 & 6) == 0 )
      {
        v17 = *(__int64 **)(v16 + 24);
        v18 = *(_DWORD *)(v16 + 32);
        if ( v18 > 0x40 )
        {
          v19 = *v17;
        }
        else
        {
          if ( !v18 )
            goto LABEL_22;
          v19 = (__int64)((_QWORD)v17 << (64 - (unsigned __int8)v18)) >> (64 - (unsigned __int8)v18);
        }
        if ( v19 < -1 )
        {
          v38 = 1;
          v9 = "invalid subrange count";
          goto LABEL_7;
        }
      }
LABEL_22:
      v20 = *(a2 - 16);
      if ( (v20 & 2) != 0 )
      {
        v21 = *((_QWORD *)a2 - 4);
        v22 = *(unsigned __int8 **)(v21 + 8);
        v23 = (const char *)*((_QWORD *)a2 - 4);
        if ( !v22 || (v24 = *v22, v24 == 1) )
        {
LABEL_28:
          v25 = *(unsigned __int8 **)(v21 + 16);
          if ( v25 )
          {
            v26 = *v25;
            if ( (unsigned __int8)v26 > 0x1Au )
              goto LABEL_57;
            goto LABEL_30;
          }
          goto LABEL_33;
        }
        if ( (unsigned int)v24 - 25 <= 1 || v24 == 7 )
        {
          v23 = (const char *)*((_QWORD *)a2 - 4);
          goto LABEL_28;
        }
LABEL_60:
        v38 = 1;
        v9 = "LowerBound must be signed constant or DIVariable or DIExpression";
        goto LABEL_7;
      }
      v23 = &v5[-8 * ((v20 >> 2) & 0xF)];
      v32 = (unsigned __int8 *)*((_QWORD *)v23 + 1);
      if ( v32 )
      {
        v33 = *v32;
        if ( v33 != 1 )
        {
          if ( (unsigned int)v33 - 25 > 1 && v33 != 7 )
            goto LABEL_60;
          v23 = &v5[-8 * ((v20 >> 2) & 0xF)];
        }
      }
      v34 = (unsigned __int8 *)*((_QWORD *)v23 + 2);
      if ( v34 )
      {
        v26 = *v34;
        if ( (unsigned __int8)v26 > 0x1Au )
          goto LABEL_57;
LABEL_30:
        v27 = 100663426;
        if ( _bittest64(&v27, v26) )
        {
          if ( (v20 & 2) != 0 )
            v23 = (const char *)*((_QWORD *)a2 - 4);
          else
            v23 = &v5[-8 * ((v20 >> 2) & 0xF)];
          goto LABEL_33;
        }
LABEL_57:
        v38 = 1;
        v9 = "UpperBound must be signed constant or DIVariable or DIExpression";
        goto LABEL_7;
      }
LABEL_33:
      result = *((_QWORD *)v23 + 3);
      if ( !result )
        return result;
      result = *(unsigned __int8 *)result;
      if ( (unsigned __int8)result <= 0x1Au )
      {
        v28 = 100663426;
        if ( _bittest64(&v28, result) )
          return result;
      }
      v38 = 1;
      v9 = "Stride must be signed constant or DIVariable or DIExpression";
LABEL_7:
      v10 = *(_QWORD *)a1;
      v36[0] = v9;
      v37 = 3;
      if ( v10 )
      {
        sub_CA0E80(v36, v10);
        v11 = *(_BYTE **)(v10 + 32);
        if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
        {
          sub_CB5D20(v10, 10);
        }
        else
        {
          *(_QWORD *)(v10 + 32) = v11 + 1;
          *v11 = 10;
        }
        v12 = *(_QWORD *)a1;
        result = (unsigned __int8)a1[154];
        a1[153] = 1;
        a1[152] |= result;
        if ( v12 )
        {
          sub_A62C00(a2, v12, (__int64)(a1 + 16), *((_QWORD *)a1 + 1));
          v14 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            return sub_CB5D20(v14, 10);
          }
          else
          {
            *(_QWORD *)(v14 + 32) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
        return result;
      }
LABEL_52:
      result = (unsigned __int8)a1[154];
      a1[153] = 1;
      a1[152] |= result;
      return result;
    }
    v38 = 1;
    v36[0] = "Subrange can have any one of count or upperBound";
    v37 = 3;
    result = sub_BDD6D0((__int64 *)a1, (__int64)v36);
    if ( !*(_QWORD *)a1 )
      return result;
    return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
  }
  v29 = *(_QWORD *)a1;
  v38 = 1;
  v36[0] = "invalid tag";
  v37 = 3;
  if ( !v29 )
    goto LABEL_52;
  sub_CA0E80(v36, v29);
  v30 = *(_BYTE **)(v29 + 32);
  if ( (unsigned __int64)v30 >= *(_QWORD *)(v29 + 24) )
  {
    sub_CB5D20(v29, 10);
  }
  else
  {
    *(_QWORD *)(v29 + 32) = v30 + 1;
    *v30 = 10;
  }
  v31 = *(_QWORD *)a1;
  result = (unsigned __int8)a1[154];
  a1[153] = 1;
  a1[152] |= result;
  if ( v31 )
    return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
  return result;
}
