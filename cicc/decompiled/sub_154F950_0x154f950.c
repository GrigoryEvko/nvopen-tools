// Function: sub_154F950
// Address: 0x154f950
//
void __fastcall sub_154F950(__int64 a1, const char *a2, size_t a3, unsigned __int8 *a4, char a5)
{
  __int64 v8; // r12
  _WORD *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax

  if ( a4 || !a5 )
  {
    v8 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
      *(_BYTE *)(a1 + 8) = 0;
    else
      v8 = sub_1263B40(*(_QWORD *)a1, *(const char **)(a1 + 16));
    v9 = *(_WORD **)(v8 + 24);
    v10 = *(_QWORD *)(v8 + 16) - (_QWORD)v9;
    if ( v10 < a3 )
    {
      v12 = sub_16E7EE0(v8, a2, a3);
      v9 = *(_WORD **)(v12 + 24);
      v8 = v12;
      v10 = *(_QWORD *)(v12 + 16) - (_QWORD)v9;
    }
    else if ( a3 )
    {
      memcpy(v9, a2, a3);
      v13 = *(_QWORD *)(v8 + 16);
      v9 = (_WORD *)(a3 + *(_QWORD *)(v8 + 24));
      *(_QWORD *)(v8 + 24) = v9;
      v10 = v13 - (_QWORD)v9;
    }
    if ( v10 <= 1 )
    {
      sub_16E7EE0(v8, ": ", 2);
      v11 = *(_QWORD *)a1;
      if ( a4 )
        goto LABEL_10;
    }
    else
    {
      *v9 = 8250;
      *(_QWORD *)(v8 + 24) += 2LL;
      v11 = *(_QWORD *)a1;
      if ( a4 )
      {
LABEL_10:
        sub_154F770(v11, a4, *(_QWORD *)(a1 + 24), *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 40));
        return;
      }
    }
    sub_1263B40(v11, "null");
  }
}
