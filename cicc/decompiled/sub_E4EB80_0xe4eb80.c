// Function: sub_E4EB80
// Address: 0xe4eb80
//
_BYTE *__fastcall sub_E4EB80(__int64 a1, __int64 a2, unsigned __int64 a3)
{
  __int64 v5; // rdi
  void *v6; // rdx
  __int64 v8; // rdi
  _BYTE *v9; // rax

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(void **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 0xAu )
  {
    sub_CB6200(v5, "\t.secrel32\t", 0xBu);
  }
  else
  {
    qmemcpy(v6, "\t.secrel32\t", 11);
    *(_QWORD *)(v5 + 32) += 11LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  if ( a3 )
  {
    v8 = *(_QWORD *)(a1 + 304);
    v9 = *(_BYTE **)(v8 + 32);
    if ( (unsigned __int64)v9 >= *(_QWORD *)(v8 + 24) )
    {
      v8 = sub_CB5D20(v8, 43);
    }
    else
    {
      *(_QWORD *)(v8 + 32) = v9 + 1;
      *v9 = 43;
    }
    sub_CB59D0(v8, a3);
  }
  return sub_E4D880(a1);
}
