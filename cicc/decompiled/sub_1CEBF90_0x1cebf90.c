// Function: sub_1CEBF90
// Address: 0x1cebf90
//
_BOOL8 __fastcall sub_1CEBF90(__int64 *a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 v6; // r12
  const char *v7; // rax
  __int64 v8; // rdx
  _BOOL8 result; // rax
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdi
  const char *v14; // rax
  __int64 v15; // rdx

  v6 = *(_QWORD *)(a2 - 24);
  result = 1;
  if ( *(_BYTE *)(v6 + 16) || (v7 = sub_1649960(*(_QWORD *)(a2 - 24)), a4 != v8) || a4 && memcmp(v7, a3, a4) )
  {
    if ( !sub_1456C80(*a1, *(_QWORD *)v6) )
      return 0;
    v10 = *a1;
    v11 = sub_146F1B0(*a1, v6);
    v12 = sub_1456F20(v10, v11);
    if ( *(_WORD *)(v12 + 24) != 10 )
      return 0;
    v13 = *(_QWORD *)(v12 - 8);
    if ( *(_BYTE *)(v13 + 16) )
      return 0;
    v14 = sub_1649960(v13);
    if ( a4 != v15 || a4 && memcmp(v14, a3, a4) )
      return 0;
  }
  return result;
}
