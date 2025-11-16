// Function: sub_E4DC10
// Address: 0xe4dc10
//
_BYTE *__fastcall sub_E4DC10(__int64 a1, __int64 a2, signed __int64 a3, unsigned int a4)
{
  __int64 v6; // rdi
  __int64 v8; // rdx
  __int64 v9; // rdi
  _WORD *v10; // rdx
  __int64 v11; // rax
  _DWORD *v12; // rdx

  v6 = *(_QWORD *)(a1 + 304);
  v8 = *(_QWORD *)(v6 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v6 + 24) - v8) <= 6 )
  {
    sub_CB6200(v6, "\t.fill\t", 7u);
  }
  else
  {
    *(_DWORD *)v8 = 1768304137;
    *(_WORD *)(v8 + 4) = 27756;
    *(_BYTE *)(v8 + 6) = 9;
    *(_QWORD *)(v6 + 32) += 7LL;
  }
  sub_E7FAD0(a2, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 312), 0);
  v9 = *(_QWORD *)(a1 + 304);
  v10 = *(_WORD **)(v9 + 32);
  if ( *(_QWORD *)(v9 + 24) - (_QWORD)v10 <= 1u )
  {
    v9 = sub_CB6200(v9, (unsigned __int8 *)", ", 2u);
  }
  else
  {
    *v10 = 8236;
    *(_QWORD *)(v9 + 32) += 2LL;
  }
  v11 = sub_CB59F0(v9, a3);
  v12 = *(_DWORD **)(v11 + 32);
  if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 3u )
  {
    sub_CB6200(v11, ", 0x", 4u);
  }
  else
  {
    *v12 = 2016419884;
    *(_QWORD *)(v11 + 32) += 4LL;
  }
  sub_CB5A50(*(_QWORD *)(a1 + 304), a4);
  return sub_E4D880(a1);
}
