// Function: sub_E81F00
// Address: 0xe81f00
//
__int64 __fastcall sub_E81F00(char *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v8; // rdx
  __int64 v10; // rcx
  char v11; // al
  unsigned __int64 v12; // rdx
  _BYTE *v13; // rdi
  __int64 result; // rax
  __int64 v15; // rdi
  unsigned int *v16; // rdi
  _BYTE *v17; // rax
  unsigned __int64 v18; // rsi
  char *v19; // rsi
  size_t v20; // rax
  size_t v21; // r14
  __int64 v22; // rdi
  __int64 v23; // rdi
  __int64 v24; // r13
  _BYTE *v25; // rax
  _DWORD *v26; // rdx

  v8 = *(void **)(a2 + 32);
  if ( *(_QWORD *)(a2 + 24) - (_QWORD)v8 <= 0xAu )
  {
    sub_CB6200(a2, "<MCOperand ", 0xBu);
    v10 = *(_QWORD *)(a2 + 32);
  }
  else
  {
    qmemcpy(v8, "<MCOperand ", 11);
    v10 = *(_QWORD *)(a2 + 32) + 11LL;
    *(_QWORD *)(a2 + 32) = v10;
  }
  v11 = *a1;
  v12 = *(_QWORD *)(a2 + 24) - v10;
  if ( !*a1 )
  {
    if ( v12 <= 6 )
    {
      sub_CB6200(a2, "INVALID", 7u);
      v13 = *(_BYTE **)(a2 + 32);
    }
    else
    {
      *(_DWORD *)v10 = 1096175177;
      *(_WORD *)(v10 + 4) = 18764;
      *(_BYTE *)(v10 + 6) = 68;
      v13 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 7LL);
      *(_QWORD *)(a2 + 32) = v13;
    }
LABEL_12:
    result = *(_QWORD *)(a2 + 24);
    goto LABEL_13;
  }
  if ( v11 == 1 )
  {
    if ( v12 <= 3 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"Reg:", 4u);
    }
    else
    {
      *(_DWORD *)v10 = 979854674;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
    v18 = *((unsigned int *)a1 + 2);
    if ( a3 )
    {
      v19 = (char *)(*(_QWORD *)(a3 + 72) + *(unsigned int *)(*(_QWORD *)(a3 + 8) + 24 * v18));
      if ( v19 )
      {
        v20 = strlen(v19);
        v13 = *(_BYTE **)(a2 + 32);
        v21 = v20;
        result = *(_QWORD *)(a2 + 24);
        if ( v21 > result - (__int64)v13 )
        {
          sub_CB6200(a2, (unsigned __int8 *)v19, v21);
          result = *(_QWORD *)(a2 + 24);
          v13 = *(_BYTE **)(a2 + 32);
        }
        else if ( v21 )
        {
          memcpy(v13, v19, v21);
          result = *(_QWORD *)(a2 + 24);
          v13 = (_BYTE *)(v21 + *(_QWORD *)(a2 + 32));
          *(_QWORD *)(a2 + 32) = v13;
        }
        goto LABEL_13;
      }
    }
    else
    {
      sub_CB59D0(a2, v18);
    }
LABEL_36:
    result = *(_QWORD *)(a2 + 24);
    v13 = *(_BYTE **)(a2 + 32);
    goto LABEL_13;
  }
  if ( v11 == 2 )
  {
    if ( v12 <= 3 )
    {
      v15 = sub_CB6200(a2, (unsigned __int8 *)"Imm:", 4u);
    }
    else
    {
      *(_DWORD *)v10 = 980249929;
      v15 = a2;
      *(_QWORD *)(a2 + 32) += 4LL;
    }
    sub_CB59F0(v15, *((_QWORD *)a1 + 1));
    result = *(_QWORD *)(a2 + 24);
    v13 = *(_BYTE **)(a2 + 32);
  }
  else if ( v11 == 3 )
  {
    if ( v12 <= 6 )
    {
      v22 = sub_CB6200(a2, "SFPImm:", 7u);
    }
    else
    {
      *(_DWORD *)v10 = 1229997651;
      *(_WORD *)(v10 + 4) = 28013;
      v22 = a2;
      *(_BYTE *)(v10 + 6) = 58;
      *(_QWORD *)(a2 + 32) += 7LL;
    }
    sub_CB5AB0(v22, *((float *)a1 + 2));
    result = *(_QWORD *)(a2 + 24);
    v13 = *(_BYTE **)(a2 + 32);
  }
  else
  {
    if ( v11 != 4 )
    {
      if ( v11 != 5 )
      {
        if ( v11 == 6 )
        {
          if ( v12 <= 5 )
          {
            sub_CB6200(a2, "Inst:(", 6u);
          }
          else
          {
            *(_DWORD *)v10 = 1953721929;
            *(_WORD *)(v10 + 4) = 10298;
            *(_QWORD *)(a2 + 32) += 6LL;
          }
          v16 = (unsigned int *)*((_QWORD *)a1 + 1);
          if ( v16 )
          {
            sub_E81DE0(v16, a2, a3);
            v17 = *(_BYTE **)(a2 + 32);
          }
          else
          {
            v26 = *(_DWORD **)(a2 + 32);
            if ( *(_QWORD *)(a2 + 24) - (_QWORD)v26 <= 3u )
            {
              sub_CB6200(a2, "NULL", 4u);
              v17 = *(_BYTE **)(a2 + 32);
            }
            else
            {
              *v26 = 1280070990;
              v17 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 4LL);
              *(_QWORD *)(a2 + 32) = v17;
            }
          }
          if ( *(_BYTE **)(a2 + 24) == v17 )
          {
            sub_CB6200(a2, (unsigned __int8 *)")", 1u);
            v13 = *(_BYTE **)(a2 + 32);
          }
          else
          {
            *v17 = 41;
            v13 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 1LL);
            *(_QWORD *)(a2 + 32) = v13;
          }
        }
        else if ( v12 <= 8 )
        {
          sub_CB6200(a2, "UNDEFINED", 9u);
          v13 = *(_BYTE **)(a2 + 32);
        }
        else
        {
          *(_BYTE *)(v10 + 8) = 68;
          *(_QWORD *)v10 = 0x454E494645444E55LL;
          v13 = (_BYTE *)(*(_QWORD *)(a2 + 32) + 9LL);
          *(_QWORD *)(a2 + 32) = v13;
        }
        goto LABEL_12;
      }
      if ( v12 <= 5 )
      {
        v24 = sub_CB6200(a2, "Expr:(", 6u);
      }
      else
      {
        *(_DWORD *)v10 = 1919973445;
        v24 = a2;
        *(_WORD *)(v10 + 4) = 10298;
        *(_QWORD *)(a2 + 32) += 6LL;
      }
      sub_E7FAD0(*((unsigned int **)a1 + 1), v24, 0, 0, a5, a6);
      v25 = *(_BYTE **)(v24 + 32);
      if ( *(_BYTE **)(v24 + 24) == v25 )
      {
        sub_CB6200(v24, (unsigned __int8 *)")", 1u);
      }
      else
      {
        *v25 = 41;
        ++*(_QWORD *)(v24 + 32);
      }
      goto LABEL_36;
    }
    if ( v12 <= 6 )
    {
      v23 = sub_CB6200(a2, "DFPImm:", 7u);
    }
    else
    {
      *(_DWORD *)v10 = 1229997636;
      v23 = a2;
      *(_WORD *)(v10 + 4) = 28013;
      *(_BYTE *)(v10 + 6) = 58;
      *(_QWORD *)(a2 + 32) += 7LL;
    }
    sub_CB5AB0(v23, *((double *)a1 + 1));
    result = *(_QWORD *)(a2 + 24);
    v13 = *(_BYTE **)(a2 + 32);
  }
LABEL_13:
  if ( (_BYTE *)result == v13 )
    return sub_CB6200(a2, (unsigned __int8 *)">", 1u);
  *v13 = 62;
  ++*(_QWORD *)(a2 + 32);
  return result;
}
