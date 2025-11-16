// Function: sub_3028140
// Address: 0x3028140
//
void __fastcall sub_3028140(_BYTE **a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v8; // r8
  __int64 v9; // rax
  _BYTE *v10; // rax
  _WORD *v11; // rdx
  unsigned int v12; // edx
  __int64 v13; // rcx

  sub_3027FE0(a1, a2, a3, a4, (__int64)a5);
  if ( a5 && !strcmp(a5, "add") )
  {
    v11 = *(_WORD **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 1u )
    {
      sub_CB6200(a4, (unsigned __int8 *)", ", 2u);
    }
    else
    {
      *v11 = 8236;
      *(_QWORD *)(a4 + 32) += 2LL;
    }
    v12 = a3 + 1;
    v13 = a4;
    goto LABEL_12;
  }
  v9 = *(_QWORD *)(a2 + 32) + 40LL * (a3 + 1);
  if ( *(_BYTE *)v9 != 1 || *(_QWORD *)(v9 + 24) )
  {
    v10 = *(_BYTE **)(a4 + 32);
    if ( *(_BYTE **)(a4 + 24) == v10 )
    {
      sub_CB6200(a4, (unsigned __int8 *)"+", 1u);
    }
    else
    {
      *v10 = 43;
      ++*(_QWORD *)(a4 + 32);
    }
    v13 = a4;
    v12 = a3 + 1;
LABEL_12:
    sub_3027FE0(a1, a2, v12, v13, v8);
  }
}
