// Function: sub_2D1FC80
// Address: 0x2d1fc80
//
bool __fastcall sub_2D1FC80(__int64 *a1, __int64 a2, const void *a3, size_t a4)
{
  __int64 v6; // r13
  const char *v7; // rax
  __int64 v8; // rdx
  __int64 v10; // r15
  __int64 *v11; // rax
  __int64 v12; // rax
  _BYTE *v13; // rdi
  const char *v14; // rax
  __int64 v15; // rdx

  v6 = *(_QWORD *)(a2 - 32);
  if ( !*(_BYTE *)v6 )
  {
    v7 = sub_BD5D20(*(_QWORD *)(a2 - 32));
    if ( v8 == a4 && (!a4 || !memcmp(v7, a3, a4)) )
      return 1;
  }
  if ( !sub_D97040(*a1, *(_QWORD *)(v6 + 8)) )
    return 0;
  v10 = *a1;
  v11 = sub_DD8400(*a1, v6);
  v12 = sub_D97190(v10, (__int64)v11);
  if ( *(_WORD *)(v12 + 24) != 15 )
    return 0;
  v13 = *(_BYTE **)(v12 - 8);
  if ( *v13 )
    return 0;
  v14 = sub_BD5D20((__int64)v13);
  if ( v15 != a4 )
    return 0;
  return !a4 || memcmp(v14, a3, a4) == 0;
}
