// Function: sub_BDD730
// Address: 0xbdd730
//
unsigned __int64 __fastcall sub_BDD730(__int64 a1, const char *a2)
{
  unsigned __int8 v4; // al
  const char **v5; // rdx
  const char *v6; // r14
  unsigned __int64 result; // rax
  __int64 v8; // r15
  _BYTE *v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rdi
  const char *v13; // rcx
  const char *v14; // rsi
  const char *v15; // r15
  __int64 v16; // r14
  _BYTE *v17; // rax
  char v18; // dl
  _QWORD v19[4]; // [rsp+0h] [rbp-50h] BYREF
  char v20; // [rsp+20h] [rbp-30h]
  char v21; // [rsp+21h] [rbp-2Fh]

  v4 = *(a2 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = (const char **)*((_QWORD *)a2 - 4);
    v6 = *v5;
    if ( !*v5 )
      goto LABEL_4;
    result = *(unsigned __int8 *)v6;
    if ( (unsigned __int8)(result - 18) > 2u )
      goto LABEL_4;
    if ( *((_DWORD *)a2 - 6) == 2 )
    {
      v15 = v5[1];
      if ( v15 )
      {
        if ( *v15 != 6 )
          goto LABEL_29;
      }
    }
LABEL_22:
    if ( (_BYTE)result == 18 && (v6[36] & 8) == 0 )
    {
      v21 = 1;
      v19[0] = "scope points into the type hierarchy";
      v20 = 3;
      result = sub_BDD6D0((__int64 *)a1, (__int64)v19);
      if ( *(_QWORD *)a1 )
        return (unsigned __int64)sub_BD9900((__int64 *)a1, a2);
    }
    return result;
  }
  v13 = a2 - 16;
  v14 = &a2[-8 * ((v4 >> 2) & 0xF) - 16];
  v6 = *(const char **)v14;
  if ( !*(_QWORD *)v14 || (unsigned __int8)(*v6 - 18) > 2u )
  {
LABEL_4:
    v8 = *(_QWORD *)a1;
    v21 = 1;
    v19[0] = "location requires a valid scope";
    v20 = 3;
    if ( v8 )
    {
      sub_CA0E80(v19, v8);
      v9 = *(_BYTE **)(v8 + 32);
      if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
      {
        sub_CB5D20(v8, 10);
      }
      else
      {
        *(_QWORD *)(v8 + 32) = v9 + 1;
        *v9 = 10;
      }
      v10 = *(_QWORD *)a1;
      result = *(unsigned __int8 *)(a1 + 154);
      *(_BYTE *)(a1 + 153) = 1;
      *(_BYTE *)(a1 + 152) |= result;
      if ( v10 )
      {
        sub_A62C00(a2, v10, a1 + 16, *(_QWORD *)(a1 + 8));
        v11 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
          result = sub_CB5D20(v11, 10);
        }
        else
        {
          *(_QWORD *)(v11 + 32) = result + 1;
          *(_BYTE *)result = 10;
        }
        if ( v6 )
        {
          sub_A62C00(v6, *(_QWORD *)a1, a1 + 16, *(_QWORD *)(a1 + 8));
          v12 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          {
            return sub_CB5D20(v12, 10);
          }
          else
          {
            *(_QWORD *)(v12 + 32) = result + 1;
            *(_BYTE *)result = 10;
          }
        }
      }
      return result;
    }
LABEL_13:
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 152) |= result;
    *(_BYTE *)(a1 + 153) = 1;
    return result;
  }
  if ( ((*((_WORD *)a2 - 8) >> 6) & 0xF) != 2 )
    goto LABEL_20;
  v15 = (const char *)*((_QWORD *)v14 + 1);
  if ( !v15 )
  {
LABEL_21:
    v6 = *(const char **)v14;
    result = **(unsigned __int8 **)v14;
    goto LABEL_22;
  }
  if ( *v15 == 6 )
  {
LABEL_20:
    v14 = &v13[-8 * ((v4 >> 2) & 0xF)];
    goto LABEL_21;
  }
LABEL_29:
  v16 = *(_QWORD *)a1;
  v21 = 1;
  v19[0] = "inlined-at should be a location";
  v20 = 3;
  if ( !v16 )
    goto LABEL_13;
  sub_CA0E80(v19, v16);
  v17 = *(_BYTE **)(v16 + 32);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(v16 + 24) )
  {
    sub_CB5D20(v16, 10);
  }
  else
  {
    *(_QWORD *)(v16 + 32) = v17 + 1;
    *v17 = 10;
  }
  result = *(_QWORD *)a1;
  v18 = *(_BYTE *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= v18;
  if ( result )
  {
    sub_BD9900((__int64 *)a1, a2);
    return (unsigned __int64)sub_BD9900((__int64 *)a1, v15);
  }
  return result;
}
