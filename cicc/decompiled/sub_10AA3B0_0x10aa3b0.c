// Function: sub_10AA3B0
// Address: 0x10aa3b0
//
bool __fastcall sub_10AA3B0(__int64 a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v5; // rax
  const void *v6; // rcx
  const void ***v7; // rdi
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // r12
  __int64 v11; // rcx
  __int64 v12; // r13
  __int16 v13; // ax
  int v14; // eax

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 != 85 )
  {
    if ( v2 != 86 )
      return 0;
    v3 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v3 != 82 )
      return 0;
    v9 = *((_QWORD *)a2 - 8);
    v10 = *(_QWORD *)(v3 - 64);
    v11 = *((_QWORD *)a2 - 4);
    v12 = *(_QWORD *)(v3 - 32);
    if ( v10 == v9 && v12 == v11 )
    {
      v13 = *(_WORD *)(v3 + 2);
    }
    else
    {
      if ( v12 != v9 || v10 != v11 )
        return 0;
      v13 = *(_WORD *)(v3 + 2);
      if ( v10 != v9 )
      {
        v14 = sub_B52870(v13 & 0x3F);
        goto LABEL_19;
      }
    }
    v14 = v13 & 0x3F;
LABEL_19:
    if ( (unsigned int)(v14 - 34) <= 1 && v10 )
    {
      **(_QWORD **)a1 = v10;
      return sub_10080A0((const void ***)(a1 + 8), v12);
    }
    return 0;
  }
  v5 = *((_QWORD *)a2 - 4);
  if ( !v5 )
    return 0;
  if ( *(_BYTE *)v5 )
    return 0;
  if ( *(_QWORD *)(v5 + 24) != *((_QWORD *)a2 + 10) )
    return 0;
  if ( (*(_BYTE *)(v5 + 33) & 0x20) == 0 )
    return 0;
  if ( *(_DWORD *)(v5 + 36) != 365 )
    return 0;
  v6 = *(const void **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
  if ( !v6 )
    return 0;
  v7 = (const void ***)(a1 + 8);
  v8 = *(_QWORD *)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
  **(v7 - 1) = v6;
  return sub_10080A0(v7, v8);
}
