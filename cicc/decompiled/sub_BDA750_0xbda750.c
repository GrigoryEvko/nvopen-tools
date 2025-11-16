// Function: sub_BDA750
// Address: 0xbda750
//
char *__fastcall sub_BDA750(__int64 a1, const char *a2)
{
  unsigned __int8 v4; // dl
  __int64 v5; // rax
  const char **v6; // rbx
  const char **v7; // rdx
  char *result; // rax
  const char *v9; // rax
  __int64 v10; // r12
  __int64 v11; // rsi
  const char *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // r14
  _QWORD v17[4]; // [rsp+0h] [rbp-50h] BYREF
  char v18; // [rsp+20h] [rbp-30h]
  char v19; // [rsp+21h] [rbp-2Fh]

  v4 = *(a2 - 16);
  if ( (v4 & 2) != 0 )
  {
    v5 = *((unsigned int *)a2 - 6);
    if ( (_DWORD)v5 )
    {
      v6 = (const char **)*((_QWORD *)a2 - 4);
LABEL_4:
      v7 = &v6[v5];
      while ( 1 )
      {
        v9 = *v6;
        if ( !*v6 )
          break;
        if ( *v9 != 1 )
          break;
        result = (char *)*((_QWORD *)v9 + 17);
        if ( *result != 17 )
          break;
        if ( v7 == ++v6 )
          return result;
      }
      v10 = *(_QWORD *)a1;
      result = "call stack metadata operand should be constant integer";
      v19 = 1;
      v17[0] = "call stack metadata operand should be constant integer";
      v18 = 3;
      if ( v10 )
      {
        sub_CA0E80(v17, v10);
        result = *(char **)(v10 + 32);
        if ( (unsigned __int64)result >= *(_QWORD *)(v10 + 24) )
        {
          result = (char *)sub_CB5D20(v10, 10);
        }
        else
        {
          *(_QWORD *)(v10 + 32) = result + 1;
          *result = 10;
        }
        v11 = *(_QWORD *)a1;
        *(_BYTE *)(a1 + 152) = 1;
        if ( v11 )
        {
          v12 = *v6;
          if ( *v6 )
          {
            v13 = *(_QWORD *)(a1 + 8);
            v14 = a1 + 16;
            goto LABEL_15;
          }
        }
        return result;
      }
LABEL_22:
      *(_BYTE *)(a1 + 152) = 1;
      return result;
    }
  }
  else
  {
    LOWORD(v5) = (*((_WORD *)a2 - 8) >> 6) & 0xF;
    if ( ((*((_WORD *)a2 - 8) >> 6) & 0xF) != 0 )
    {
      v5 = (unsigned __int8)v5;
      v6 = (const char **)&a2[-8 * ((v4 >> 2) & 0xF) - 16];
      goto LABEL_4;
    }
  }
  v16 = *(_QWORD *)a1;
  result = "call stack metadata should have at least 1 operand";
  v19 = 1;
  v17[0] = "call stack metadata should have at least 1 operand";
  v18 = 3;
  if ( !v16 )
    goto LABEL_22;
  sub_CA0E80(v17, v16);
  result = *(char **)(v16 + 32);
  if ( (unsigned __int64)result >= *(_QWORD *)(v16 + 24) )
  {
    result = (char *)sub_CB5D20(v16, 10);
  }
  else
  {
    *(_QWORD *)(v16 + 32) = result + 1;
    *result = 10;
  }
  v11 = *(_QWORD *)a1;
  *(_BYTE *)(a1 + 152) = 1;
  if ( v11 )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v14 = a1 + 16;
    v12 = a2;
LABEL_15:
    sub_A62C00(v12, v11, v14, v13);
    v15 = *(_QWORD *)a1;
    result = *(char **)(*(_QWORD *)a1 + 32LL);
    if ( (unsigned __int64)result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
    {
      return (char *)sub_CB5D20(v15, 10);
    }
    else
    {
      *(_QWORD *)(v15 + 32) = result + 1;
      *result = 10;
    }
  }
  return result;
}
