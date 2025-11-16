// Function: sub_E4F4D0
// Address: 0xe4f4d0
//
_BYTE *__fastcall sub_E4F4D0(__int64 a1, __int64 a2, unsigned __int64 a3, unsigned __int8 a4)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _BYTE *v10; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rax
  _BYTE *result; // rax

  v6 = *(_QWORD *)(a1 + 304);
  v8 = *(_QWORD *)(v6 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v8) <= 6 )
  {
    sub_CB6200(v6, "\t.comm\t", 7u);
  }
  else
  {
    *(_DWORD *)v8 = 1868770825;
    *(_WORD *)(v8 + 4) = 28013;
    *(_BYTE *)(v8 + 6) = 9;
    *(_QWORD *)(v6 + 32) += 7LL;
  }
  sub_EA12C0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312));
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_BYTE **)(v9 + 32);
  if ( (unsigned __int64)v10 >= *(_QWORD *)(v9 + 24) )
  {
    v9 = sub_CB5D20(v9, 44);
  }
  else
  {
    *(_QWORD *)(v9 + 32) = v10 + 1;
    *v10 = 44;
  }
  sub_CB59D0(v9, a3);
  v11 = *(_QWORD *)(a1 + 304);
  v12 = *(_BYTE **)(v11 + 32);
  if ( *(_BYTE *)(*(_QWORD *)(a1 + 312) + 281LL) )
  {
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
    {
      v11 = sub_CB5D20(v11, 44);
    }
    else
    {
      *(_QWORD *)(v11 + 32) = v12 + 1;
      *v12 = 44;
    }
    sub_CB59D0(v11, 1LL << a4);
  }
  else
  {
    if ( (unsigned __int64)v12 >= *(_QWORD *)(v11 + 24) )
    {
      v11 = sub_CB5D20(v11, 44);
    }
    else
    {
      *(_QWORD *)(v11 + 32) = v12 + 1;
      *v12 = 44;
    }
    sub_CB59D0(v11, a4);
  }
  sub_E4D880(a1);
  result = (_BYTE *)(*(_BYTE *)(a2 + 9) & 7);
  if ( (_BYTE)result == 6 )
  {
    if ( *(_BYTE *)(a2 + 72) )
      return sub_E4E8B0(a1, a2, *(char **)(a2 + 56), *(_QWORD *)(a2 + 64));
  }
  return result;
}
