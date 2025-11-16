// Function: sub_E4EA70
// Address: 0xe4ea70
//
_BYTE *__fastcall sub_E4EA70(__int64 a1, __int64 a2, signed __int64 a3)
{
  __int64 v5; // rdi
  __int64 v6; // rdx
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // rdi
  _BYTE *v11; // rax

  v5 = *(_QWORD *)(a1 + 304);
  v6 = *(_QWORD *)(v5 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v5 + 24) - v6) <= 5 )
  {
    sub_CB6200(v5, "\t.rva\t", 6u);
  }
  else
  {
    *(_DWORD *)v6 = 1987194377;
    *(_WORD *)(v6 + 4) = 2401;
    *(_QWORD *)(v5 + 32) += 6LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  if ( a3 > 0 )
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
    sub_CB59F0(v8, a3);
  }
  else if ( a3 )
  {
    v10 = *(_QWORD *)(a1 + 304);
    v11 = *(_BYTE **)(v10 + 32);
    if ( (unsigned __int64)v11 >= *(_QWORD *)(v10 + 24) )
    {
      v10 = sub_CB5D20(v10, 45);
    }
    else
    {
      *(_QWORD *)(v10 + 32) = v11 + 1;
      *v11 = 45;
    }
    sub_CB59F0(v10, -a3);
  }
  return sub_E4D880(a1);
}
