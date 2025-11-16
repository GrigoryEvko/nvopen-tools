// Function: sub_1643DA0
// Address: 0x1643da0
//
bool __fastcall sub_1643DA0(__int64 a1, _BYTE *a2)
{
  __int64 v3; // rdi
  _BYTE *v4; // r12
  __int64 v5; // rcx
  unsigned __int8 v6; // al
  _QWORD *v7; // rax
  __int64 v9; // rax
  char v10; // al

  v3 = *(_QWORD *)a2;
  if ( *(_BYTE *)(a1 + 8) == 13 )
  {
    v4 = a2;
    if ( *(_BYTE *)(v3 + 8) == 16 )
    {
      if ( !sub_1642F90(**(_QWORD **)(v3 + 16), 32) )
        return 0;
    }
    else if ( !sub_1642F90(v3, 32) )
    {
      return 0;
    }
    v6 = a2[16];
    if ( v6 <= 0x10u )
    {
      if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
      {
        v9 = sub_15A1020(a2, 32, *(_QWORD *)a2, v5);
        v4 = (_BYTE *)v9;
        if ( !v9 )
          return 0;
        v6 = *(_BYTE *)(v9 + 16);
      }
      if ( v6 == 13 )
      {
        v7 = (_QWORD *)*((_QWORD *)v4 + 3);
        if ( *((_DWORD *)v4 + 8) > 0x40u )
          v7 = (_QWORD *)*v7;
        return *(unsigned int *)(a1 + 12) > (unsigned __int64)v7;
      }
    }
    return 0;
  }
  v10 = *(_BYTE *)(v3 + 8);
  if ( v10 == 16 )
    v10 = *(_BYTE *)(**(_QWORD **)(v3 + 16) + 8LL);
  return v10 == 11;
}
