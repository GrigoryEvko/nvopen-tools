// Function: sub_10C89E0
// Address: 0x10c89e0
//
bool __fastcall sub_10C89E0(__int64 a1, int a2, unsigned __int8 *a3)
{
  char v5; // al
  _BYTE *v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rcx
  _BYTE *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rax

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = sub_995B10((_QWORD **)a1, *((_QWORD *)a3 - 8));
  v6 = (_BYTE *)*((_QWORD *)a3 - 4);
  if ( v5 && *v6 == 58 )
  {
    v7 = *((_QWORD *)v6 - 8);
    v8 = *(_QWORD *)(a1 + 8);
    v9 = *((_QWORD *)v6 - 4);
    if ( v7 == v8 && v9 == *(_QWORD *)(a1 + 16) )
      return 1;
    if ( v8 == v9 && v7 == *(_QWORD *)(a1 + 16) )
      return 1;
  }
  if ( (unsigned __int8)sub_995B10((_QWORD **)a1, (__int64)v6) )
  {
    v10 = (_BYTE *)*((_QWORD *)a3 - 8);
    if ( *v10 == 58 )
    {
      v11 = *((_QWORD *)v10 - 8);
      v12 = *(_QWORD *)(a1 + 8);
      v13 = *((_QWORD *)v10 - 4);
      if ( v11 == v12 && v13 == *(_QWORD *)(a1 + 16) )
        return 1;
      if ( v13 == v12 )
        return *(_QWORD *)(a1 + 16) == v11;
    }
  }
  return 0;
}
