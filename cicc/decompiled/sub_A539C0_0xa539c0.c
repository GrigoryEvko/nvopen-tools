// Function: sub_A539C0
// Address: 0xa539c0
//
void __fastcall sub_A539C0(__int64 a1, const void *a2, size_t a3, __int64 a4)
{
  __int64 v6; // r12
  _WORD *v7; // rdi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax

  if ( a4 )
  {
    v6 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 8) )
      *(_BYTE *)(a1 + 8) = 0;
    else
      v6 = sub_904010(*(_QWORD *)a1, *(const char **)(a1 + 16));
    v7 = *(_WORD **)(v6 + 32);
    v8 = *(_QWORD *)(v6 + 24) - (_QWORD)v7;
    if ( v8 < a3 )
    {
      v9 = sub_CB6200(v6, a2, a3);
      v7 = *(_WORD **)(v9 + 32);
      v6 = v9;
      v8 = *(_QWORD *)(v9 + 24) - (_QWORD)v7;
    }
    else if ( a3 )
    {
      memcpy(v7, a2, a3);
      v10 = *(_QWORD *)(v6 + 24);
      v7 = (_WORD *)(a3 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v7;
      v8 = v10 - (_QWORD)v7;
    }
    if ( v8 <= 1 )
    {
      v6 = sub_CB6200(v6, ": ", 2);
    }
    else
    {
      *v7 = 8250;
      *(_QWORD *)(v6 + 32) += 2LL;
    }
    sub_CB59D0(v6, a4);
  }
}
