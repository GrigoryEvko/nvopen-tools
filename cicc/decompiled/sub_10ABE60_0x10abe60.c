// Function: sub_10ABE60
// Address: 0x10abe60
//
__int64 __fastcall sub_10ABE60(unsigned __int8 ***a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v5; // rax
  unsigned __int8 *v6; // r12
  unsigned __int8 *v7; // r13
  unsigned __int8 *v8; // rdx
  unsigned __int8 *v9; // rcx
  __int16 v10; // ax
  int v11; // eax

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
      || *(_DWORD *)(v5 + 36) != 366 )
    {
      return 0;
    }
    v6 = *(unsigned __int8 **)&a2[-32 * (*((_DWORD *)a2 + 1) & 0x7FFFFFF)];
    v7 = *(unsigned __int8 **)&a2[32 * (1LL - (*((_DWORD *)a2 + 1) & 0x7FFFFFF))];
    if ( !v6 )
      goto LABEL_21;
    goto LABEL_13;
  }
  if ( v2 != 86 )
    return 0;
  v3 = *((_QWORD *)a2 - 12);
  if ( *(_BYTE *)v3 != 82 )
    return 0;
  v8 = (unsigned __int8 *)*((_QWORD *)a2 - 8);
  v6 = *(unsigned __int8 **)(v3 - 64);
  v9 = (unsigned __int8 *)*((_QWORD *)a2 - 4);
  v7 = *(unsigned __int8 **)(v3 - 32);
  if ( v6 == v8 && v7 == v9 )
  {
    v10 = *(_WORD *)(v3 + 2);
LABEL_18:
    v11 = v10 & 0x3F;
    goto LABEL_19;
  }
  if ( v7 != v8 || v6 != v9 )
    return 0;
  v10 = *(_WORD *)(v3 + 2);
  if ( v6 == v8 )
    goto LABEL_18;
  v11 = sub_B52870(v10 & 0x3F);
LABEL_19:
  if ( (unsigned int)(v11 - 36) > 1 )
    return 0;
  if ( v6 )
  {
LABEL_13:
    **a1 = v6;
    if ( (unsigned __int8)sub_996420(a1 + 1, 30, v7) )
      return 1;
  }
LABEL_21:
  if ( !v7 )
    return 0;
  **a1 = v7;
  return sub_996420(a1 + 1, 30, v6);
}
