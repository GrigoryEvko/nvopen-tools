// Function: sub_BDDA40
// Address: 0xbdda40
//
unsigned __int64 __fastcall sub_BDDA40(__int64 a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char *v5; // r14
  unsigned __int64 result; // rax
  __int64 v7; // r15
  _BYTE *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdx
  const char *v14; // rdi
  __int64 v15; // r14
  _BYTE *v16; // rax
  __int64 v17; // rdi
  _QWORD v18[4]; // [rsp+0h] [rbp-50h] BYREF
  char v19; // [rsp+20h] [rbp-30h]
  char v20; // [rsp+21h] [rbp-2Fh]

  if ( (unsigned __int16)sub_AF18C0((__int64)a2) == 11 )
  {
    v4 = *(a2 - 16);
    if ( (v4 & 2) != 0 )
    {
      v5 = *(const char **)(*((_QWORD *)a2 - 4) + 8LL);
      if ( !v5 || (result = *(unsigned __int8 *)v5, (unsigned __int8)(result - 18) > 2u) )
      {
LABEL_5:
        v7 = *(_QWORD *)a1;
        v20 = 1;
        v18[0] = "invalid local scope";
        v19 = 3;
        if ( v7 )
        {
          sub_CA0E80(v18, v7);
          v8 = *(_BYTE **)(v7 + 32);
          if ( (unsigned __int64)v8 >= *(_QWORD *)(v7 + 24) )
          {
            sub_CB5D20(v7, 10);
          }
          else
          {
            *(_QWORD *)(v7 + 32) = v8 + 1;
            *v8 = 10;
          }
          v9 = *(_QWORD *)a1;
          result = *(unsigned __int8 *)(a1 + 154);
          *(_BYTE *)(a1 + 153) = 1;
          *(_BYTE *)(a1 + 152) |= result;
          if ( v9 )
          {
            sub_A62C00(a2, v9, a1 + 16, *(_QWORD *)(a1 + 8));
            v10 = *(_QWORD *)a1;
            result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
            if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            {
              result = sub_CB5D20(v10, 10);
            }
            else
            {
              *(_QWORD *)(v10 + 32) = result + 1;
              *(_BYTE *)result = 10;
            }
            if ( v5 )
            {
              v11 = *(_QWORD *)(a1 + 8);
              v12 = *(_QWORD *)a1;
              v13 = a1 + 16;
              v14 = v5;
              goto LABEL_26;
            }
          }
          return result;
        }
LABEL_20:
        result = *(unsigned __int8 *)(a1 + 154);
        *(_BYTE *)(a1 + 152) |= result;
        *(_BYTE *)(a1 + 153) = 1;
        return result;
      }
    }
    else
    {
      v5 = *(const char **)&a2[-8 * ((v4 >> 2) & 0xF) - 8];
      if ( !v5 )
        goto LABEL_5;
      result = *(unsigned __int8 *)v5;
      if ( (unsigned __int8)(result - 18) > 2u )
        goto LABEL_5;
    }
    if ( (_BYTE)result == 18 && (v5[36] & 8) == 0 )
    {
      v20 = 1;
      v18[0] = "scope points into the type hierarchy";
      v19 = 3;
      result = sub_BDD6D0((__int64 *)a1, (__int64)v18);
      if ( *(_QWORD *)a1 )
        return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
    }
    return result;
  }
  v15 = *(_QWORD *)a1;
  v20 = 1;
  v18[0] = "invalid tag";
  v19 = 3;
  if ( !v15 )
    goto LABEL_20;
  sub_CA0E80(v18, v15);
  v16 = *(_BYTE **)(v15 + 32);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v15 + 24) )
  {
    sub_CB5D20(v15, 10);
  }
  else
  {
    *(_QWORD *)(v15 + 32) = v16 + 1;
    *v16 = 10;
  }
  v12 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= result;
  if ( v12 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v13 = a1 + 16;
    v14 = a2;
LABEL_26:
    sub_A62C00(v14, v12, v13, v11);
    v17 = *(_QWORD *)a1;
    result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
    if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
    {
      return sub_CB5D20(v17, 10);
    }
    else
    {
      *(_QWORD *)(v17 + 32) = result + 1;
      *(_BYTE *)result = 10;
    }
  }
  return result;
}
