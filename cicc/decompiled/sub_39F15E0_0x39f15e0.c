// Function: sub_39F15E0
// Address: 0x39f15e0
//
_BYTE *__fastcall sub_39F15E0(char *a1, __int64 a2)
{
  void *v3; // rdx
  __int64 v4; // rcx
  char v5; // al
  unsigned __int64 v6; // rdx
  _BYTE *result; // rax
  __int64 v8; // rdi
  __int64 v9; // r13
  _BYTE *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi

  v3 = *(void **)(a2 + 24);
  if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0xAu )
  {
    sub_16E7EE0(a2, "<MCOperand ", 0xBu);
    v4 = *(_QWORD *)(a2 + 24);
  }
  else
  {
    qmemcpy(v3, "<MCOperand ", 11);
    v4 = *(_QWORD *)(a2 + 24) + 11LL;
    *(_QWORD *)(a2 + 24) = v4;
  }
  v5 = *a1;
  v6 = *(_QWORD *)(a2 + 16) - v4;
  if ( !*a1 )
  {
    if ( v6 <= 6 )
    {
      sub_16E7EE0(a2, "INVALID", 7u);
      result = *(_BYTE **)(a2 + 24);
    }
    else
    {
      *(_DWORD *)v4 = 1096175177;
      *(_WORD *)(v4 + 4) = 18764;
      *(_BYTE *)(v4 + 6) = 68;
      result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 7LL);
      *(_QWORD *)(a2 + 24) = result;
    }
    goto LABEL_11;
  }
  if ( v5 == 1 )
  {
    if ( v6 <= 3 )
    {
      v11 = sub_16E7EE0(a2, "Reg:", 4u);
    }
    else
    {
      *(_DWORD *)v4 = 979854674;
      v11 = a2;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    sub_16E7A90(v11, *((unsigned int *)a1 + 2));
    result = *(_BYTE **)(a2 + 24);
  }
  else if ( v5 == 2 )
  {
    if ( v6 <= 3 )
    {
      v8 = sub_16E7EE0(a2, "Imm:", 4u);
    }
    else
    {
      *(_DWORD *)v4 = 980249929;
      v8 = a2;
      *(_QWORD *)(a2 + 24) += 4LL;
    }
    sub_16E7AB0(v8, *((_QWORD *)a1 + 1));
    result = *(_BYTE **)(a2 + 24);
  }
  else
  {
    if ( v5 != 3 )
    {
      if ( v5 == 4 )
      {
        if ( v6 <= 5 )
        {
          v9 = sub_16E7EE0(a2, "Expr:(", 6u);
        }
        else
        {
          *(_DWORD *)v4 = 1919973445;
          v9 = a2;
          *(_WORD *)(v4 + 4) = 10298;
          *(_QWORD *)(a2 + 24) += 6LL;
        }
        sub_38CDBE0(*((_QWORD *)a1 + 1), v9, 0);
        v10 = *(_BYTE **)(v9 + 24);
        if ( *(_BYTE **)(v9 + 16) != v10 )
        {
LABEL_23:
          *v10 = 41;
          ++*(_QWORD *)(v9 + 24);
LABEL_24:
          result = *(_BYTE **)(a2 + 24);
          goto LABEL_11;
        }
      }
      else
      {
        if ( v5 != 5 )
        {
          if ( v6 > 8 )
          {
            *(_BYTE *)(v4 + 8) = 68;
            *(_QWORD *)v4 = 0x454E494645444E55LL;
            result = (_BYTE *)(*(_QWORD *)(a2 + 24) + 9LL);
            *(_QWORD *)(a2 + 24) = result;
            goto LABEL_11;
          }
          sub_16E7EE0(a2, "UNDEFINED", 9u);
          goto LABEL_24;
        }
        if ( v6 <= 5 )
        {
          v9 = sub_16E7EE0(a2, "Inst:(", 6u);
        }
        else
        {
          *(_DWORD *)v4 = 1953721929;
          v9 = a2;
          *(_WORD *)(v4 + 4) = 10298;
          *(_QWORD *)(a2 + 24) += 6LL;
        }
        sub_39F14C0(*((unsigned int **)a1 + 1), v9);
        v10 = *(_BYTE **)(v9 + 24);
        if ( *(_BYTE **)(v9 + 16) != v10 )
          goto LABEL_23;
      }
      sub_16E7EE0(v9, ")", 1u);
      goto LABEL_24;
    }
    if ( v6 <= 5 )
    {
      v12 = sub_16E7EE0(a2, "FPImm:", 6u);
    }
    else
    {
      *(_DWORD *)v4 = 1833521222;
      v12 = a2;
      *(_WORD *)(v4 + 4) = 14957;
      *(_QWORD *)(a2 + 24) += 6LL;
    }
    sub_16E7B70(v12);
    result = *(_BYTE **)(a2 + 24);
  }
LABEL_11:
  if ( result == *(_BYTE **)(a2 + 16) )
    return (_BYTE *)sub_16E7EE0(a2, ">", 1u);
  *result = 62;
  ++*(_QWORD *)(a2 + 24);
  return result;
}
