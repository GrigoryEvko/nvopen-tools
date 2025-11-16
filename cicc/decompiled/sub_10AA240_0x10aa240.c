// Function: sub_10AA240
// Address: 0x10aa240
//
bool __fastcall sub_10AA240(_QWORD *a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r13
  __int16 v12; // ax
  int v13; // eax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v5 = *((_QWORD *)a2 - 4);
    if ( !v5
      || *(_BYTE *)v5
      || *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10)
      || (*(_BYTE *)(v5 + 33) & 0x20) == 0
      || *(_DWORD *)(v5 + 36) != 330 )
    {
      return 0;
    }
    v6 = *(_QWORD *)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v7 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( v6 != *a1 || v7 != a1[1] )
    {
      if ( v7 == *a1 )
        return a1[1] == v6;
      return 0;
    }
    return 1;
  }
  if ( v2 != 86 )
    return 0;
  v3 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v3 != 82 )
    return 0;
  v8 = *((_QWORD *)a2 - 8);
  v9 = *(_QWORD *)(v3 - 64);
  v10 = *((_QWORD *)a2 - 4);
  v11 = *(_QWORD *)(v3 - 32);
  if ( v9 == v8 && v11 == v10 )
  {
    v12 = *(_WORD *)(v3 + 2);
LABEL_19:
    v13 = v12 & 0x3F;
    goto LABEL_20;
  }
  if ( v11 != v8 || v9 != v10 )
    return 0;
  v12 = *(_WORD *)(v3 + 2);
  if ( v9 == v8 )
    goto LABEL_19;
  v13 = sub_B52870(v12 & 0x3F);
LABEL_20:
  if ( (unsigned int)(v13 - 40) <= 1 )
  {
    if ( v9 != *a1 || v11 != a1[1] )
    {
      if ( v11 == *a1 )
        return a1[1] == v9;
      return 0;
    }
    return 1;
  }
  return 0;
}
