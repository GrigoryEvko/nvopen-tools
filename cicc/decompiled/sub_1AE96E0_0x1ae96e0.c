// Function: sub_1AE96E0
// Address: 0x1ae96e0
//
char __fastcall sub_1AE96E0(__int64 a1, __int64 *a2)
{
  int v2; // eax
  unsigned __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v12; // rax
  char v13; // dl
  __int64 v14; // rdx
  int v15; // eax
  __int64 v16; // rax
  unsigned int v17; // ebx
  bool v18; // al
  unsigned __int8 v19; // al
  __int64 v20; // rdi

  v2 = *(unsigned __int8 *)(a1 + 16);
  if ( (unsigned int)(v2 - 25) <= 9 )
    return 0;
  v3 = (unsigned int)(v2 - 34);
  if ( (unsigned int)v3 <= 0x36 )
  {
    v4 = 0x40018000000001LL;
    if ( _bittest64(&v4, v3) )
      return 0;
  }
  if ( *(_BYTE *)(a1 + 16) == 78 )
  {
    v12 = *(_QWORD *)(a1 - 24);
    if ( !*(_BYTE *)(v12 + 16) )
    {
      v13 = *(_BYTE *)(v12 + 33);
      if ( (v13 & 0x20) != 0 )
      {
        if ( *(_DWORD *)(v12 + 36) == 36 )
          return sub_1601A30(a1, 1) == 0;
        if ( (v13 & 0x20) != 0 )
        {
          if ( *(_DWORD *)(v12 + 36) == 38 )
            return sub_1601A30(a1, 0) == 0;
          if ( (v13 & 0x20) != 0 && *(_DWORD *)(v12 + 36) == 37 )
            return *(_QWORD *)(*(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF)) + 24LL) == 0;
        }
      }
    }
  }
  if ( !(unsigned __int8)sub_15F3040(a1) && !sub_15F3330(a1) )
    return 1;
  if ( *(_BYTE *)(a1 + 16) != 78
    || (v14 = *(_QWORD *)(a1 - 24), *(_BYTE *)(v14 + 16))
    || (*(_BYTE *)(v14 + 33) & 0x20) == 0 )
  {
LABEL_7:
    if ( !(unsigned __int8)sub_140B1C0(a1, a2, 0) )
    {
      v5 = sub_140B650(a1, a2);
      v6 = v5;
      if ( !v5
        || (v7 = *(_DWORD *)(v5 + 20) & 0xFFFFFFF,
            v8 = 4 * v7,
            v9 = -3 * v7,
            v10 = *(_QWORD *)(v6 + 8 * v9),
            *(_BYTE *)(v10 + 16) > 0x10u) )
      {
        v19 = *(_BYTE *)(a1 + 16);
        if ( v19 > 0x17u )
        {
          if ( v19 == 78 )
          {
            v20 = a1 | 4;
          }
          else
          {
            if ( v19 != 29 )
              return 0;
            v20 = a1 & 0xFFFFFFFFFFFFFFFBLL;
          }
          if ( (v20 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            return sub_14DA760(v20, a2);
        }
        return 0;
      }
      if ( !sub_1593BB0(*(_QWORD *)(v6 + 8 * v9), (__int64)a2, v6, v8) )
        return *(_BYTE *)(v10 + 16) == 9;
    }
    return 1;
  }
  v15 = *(_DWORD *)(v14 + 36);
  if ( v15 == 115 || v15 == 202 )
    return 1;
  if ( (unsigned int)(v15 - 116) <= 1 )
    return *(_BYTE *)(*(_QWORD *)(a1 + 24 * (1LL - (*(_DWORD *)(a1 + 20) & 0xFFFFFFF))) + 16LL) == 9;
  if ( v15 != 4 )
  {
LABEL_32:
    if ( *(_DWORD *)(v14 + 36) == 79 )
      goto LABEL_33;
    goto LABEL_7;
  }
  if ( !sub_1602380(a1) )
  {
    v14 = *(_QWORD *)(a1 - 24);
    if ( *(_BYTE *)(v14 + 16) )
      BUG();
    goto LABEL_32;
  }
LABEL_33:
  v16 = *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF));
  if ( *(_BYTE *)(v16 + 16) != 13 )
    return 0;
  v17 = *(_DWORD *)(v16 + 32);
  if ( v17 <= 0x40 )
    v18 = *(_QWORD *)(v16 + 24) == 0;
  else
    v18 = v17 == (unsigned int)sub_16A57B0(v16 + 24);
  return !v18;
}
