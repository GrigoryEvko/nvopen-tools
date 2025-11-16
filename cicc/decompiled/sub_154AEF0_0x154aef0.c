// Function: sub_154AEF0
// Address: 0x154aef0
//
void __fastcall sub_154AEF0(__int64 a1, const char *a2, size_t a3, __int64 a4, char a5)
{
  __int64 v7; // r12
  _WORD *v8; // rdi
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax

  if ( a4 || !a5 )
  {
    v7 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
      *(_BYTE *)(a1 + 8) = 0;
    else
      v7 = sub_1263B40(*(_QWORD *)a1, *(const char **)(a1 + 16));
    v8 = *(_WORD **)(v7 + 24);
    v9 = *(_QWORD *)(v7 + 16) - (_QWORD)v8;
    if ( v9 < a3 )
    {
      v10 = sub_16E7EE0(v7, a2, a3);
      v8 = *(_WORD **)(v10 + 24);
      v7 = v10;
      v9 = *(_QWORD *)(v10 + 16) - (_QWORD)v8;
    }
    else if ( a3 )
    {
      memcpy(v8, a2, a3);
      v11 = *(_QWORD *)(v7 + 16);
      v8 = (_WORD *)(a3 + *(_QWORD *)(v7 + 24));
      *(_QWORD *)(v7 + 24) = v8;
      v9 = v11 - (_QWORD)v8;
    }
    if ( v9 <= 1 )
    {
      v7 = sub_16E7EE0(v7, ": ", 2);
    }
    else
    {
      *v8 = 8250;
      *(_QWORD *)(v7 + 24) += 2LL;
    }
    sub_16E7AB0(v7, a4);
  }
}
