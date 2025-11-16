// Function: sub_21544A0
// Address: 0x21544a0
//
char __fastcall sub_21544A0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4, const char *a5)
{
  __int64 v8; // rax
  _BYTE *v9; // rax
  _WORD *v10; // rdx
  unsigned int v11; // edx
  __int64 v12; // rcx

  sub_2154370(a1, a2, a3, a4, 0);
  if ( a5 && !strcmp(a5, "add") )
  {
    v10 = *(_WORD **)(a4 + 24);
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)v10 <= 1u )
    {
      sub_16E7EE0(a4, ", ", 2u);
    }
    else
    {
      *v10 = 8236;
      *(_QWORD *)(a4 + 24) += 2LL;
    }
    v11 = a3 + 1;
    v12 = a4;
    goto LABEL_12;
  }
  v8 = *(_QWORD *)(a2 + 32) + 40LL * (a3 + 1);
  if ( *(_BYTE *)v8 != 1 || *(_QWORD *)(v8 + 24) )
  {
    v9 = *(_BYTE **)(a4 + 24);
    if ( *(_BYTE **)(a4 + 16) == v9 )
    {
      sub_16E7EE0(a4, "+", 1u);
    }
    else
    {
      *v9 = 43;
      ++*(_QWORD *)(a4 + 24);
    }
    v12 = a4;
    v11 = a3 + 1;
LABEL_12:
    LOBYTE(v8) = sub_2154370(a1, a2, v11, v12, 0);
  }
  return v8;
}
