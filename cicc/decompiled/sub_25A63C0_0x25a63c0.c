// Function: sub_25A63C0
// Address: 0x25a63c0
//
__int64 __fastcall sub_25A63C0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  char *v5; // rcx
  const void *v6; // rax
  const void *v7; // rsi
  __int64 v8; // rbx

  *(_DWORD *)a1 = *(_DWORD *)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
      sub_4261EA(a1, a2, a3);
    v5 = (char *)sub_22077B0(v4);
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  *(_QWORD *)(a1 + 8) = v5;
  *(_QWORD *)(a1 + 24) = &v5[v4];
  *(_QWORD *)(a1 + 16) = v5;
  v6 = *(const void **)(a2 + 56);
  v7 = *(const void **)(a2 + 48);
  v8 = *(_QWORD *)(a2 + 56) - (_QWORD)v7;
  if ( v6 != v7 )
    v5 = (char *)memmove(v5, v7, *(_QWORD *)(a2 + 56) - (_QWORD)v7);
  *(_QWORD *)(a1 + 16) = &v5[v8];
  return a1;
}
