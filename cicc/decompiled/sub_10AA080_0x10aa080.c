// Function: sub_10AA080
// Address: 0x10aa080
//
bool __fastcall sub_10AA080(__int64 a1, char *a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r13
  __int16 v11; // ax
  int v12; // eax
  __int64 v13; // rax

  v13 = *((_QWORD *)a2 + 2);
  if ( !v13 || *(_QWORD *)(v13 + 8) )
    return 0;
  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v4 = *((_QWORD *)a2 - 4);
    if ( !v4
      || *(_BYTE *)v4
      || *(_QWORD *)(v4 + 24) != *((_QWORD *)a2 + 10)
      || (*(_BYTE *)(v4 + 33) & 0x20) == 0
      || *(_DWORD *)(v4 + 36) != 366 )
    {
      return 0;
    }
    v5 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v6 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( !v5 || (**(_QWORD **)a1 = v5, v6 != *(_QWORD *)(a1 + 8)) )
    {
      if ( v6 )
      {
        **(_QWORD **)a1 = v6;
        return *(_QWORD *)(a1 + 8) == v5;
      }
      return 0;
    }
    return 1;
  }
  if ( v2 != 86 )
    return 0;
  v3 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v3 != 82 )
    return 0;
  v7 = *((_QWORD *)a2 - 8);
  v8 = *(_QWORD *)(v3 - 64);
  v9 = *((_QWORD *)a2 - 4);
  v10 = *(_QWORD *)(v3 - 32);
  if ( v8 == v7 && v10 == v9 )
  {
    v11 = *(_WORD *)(v3 + 2);
LABEL_23:
    v12 = v11 & 0x3F;
    goto LABEL_24;
  }
  if ( v10 != v7 || v8 != v9 )
    return 0;
  v11 = *(_WORD *)(v3 + 2);
  if ( v8 == v7 )
    goto LABEL_23;
  v12 = sub_B52870(v11 & 0x3F);
LABEL_24:
  if ( (unsigned int)(v12 - 36) <= 1 )
  {
    if ( !v8 || (**(_QWORD **)a1 = v8, v10 != *(_QWORD *)(a1 + 8)) )
    {
      if ( v10 )
      {
        **(_QWORD **)a1 = v10;
        return *(_QWORD *)(a1 + 8) == v8;
      }
      return 0;
    }
    return 1;
  }
  return 0;
}
