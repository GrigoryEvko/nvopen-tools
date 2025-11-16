// Function: sub_BCED00
// Address: 0xbced00
//
_BOOL8 __fastcall sub_BCED00(__int64 a1, __int64 a2)
{
  _BOOL4 v2; // r13d
  __int64 *v3; // rax
  __int64 v4; // rcx
  __int64 *v5; // rdx
  _BOOL4 v6; // eax
  __int64 *v7; // rbx
  __int64 *v8; // r13
  __int64 v9; // r15
  int v10; // edx
  char v12; // dl
  _BOOL4 v13; // eax

  v2 = 1;
  if ( (*(_DWORD *)(a1 + 8) & 0x800) == 0 )
  {
    v2 = (*(_DWORD *)(a1 + 8) & 0x100) == 0;
    if ( (*(_DWORD *)(a1 + 8) & 0x100) == 0 )
      return 0;
    if ( !a2 )
    {
LABEL_8:
      LOBYTE(v6) = sub_BCB460(a1);
      v2 = v6;
      if ( !v6 )
      {
LABEL_9:
        v7 = *(__int64 **)(a1 + 16);
        v8 = &v7[*(unsigned int *)(a1 + 12)];
        if ( v7 == v8 )
        {
LABEL_29:
          *(_DWORD *)(a1 + 8) |= 0x800u;
          return 1;
        }
        while ( 1 )
        {
          v9 = *v7;
          if ( sub_BCEA30(*v7) )
            return 0;
          v10 = *(unsigned __int8 *)(v9 + 8);
          if ( (_BYTE)v10 != 12
            && (unsigned __int8)v10 > 3u
            && (_BYTE)v10 != 5
            && (v10 & 0xFD) != 4
            && (v10 & 0xFB) != 0xA
            && ((unsigned __int8)(*(_BYTE *)(v9 + 8) - 15) > 3u && v10 != 20 || !(unsigned __int8)sub_BCEBA0(v9, a2)) )
          {
            return 0;
          }
          if ( v8 == ++v7 )
            goto LABEL_29;
        }
      }
LABEL_28:
      *(_DWORD *)(a1 + 8) |= 0x800u;
      return v2;
    }
    if ( !*(_BYTE *)(a2 + 28) )
    {
LABEL_26:
      sub_C8CC70(a2, a1);
      if ( !v12 )
        return v2;
      LOBYTE(v13) = sub_BCB460(a1);
      v2 = v13;
      if ( !v13 )
        goto LABEL_9;
      goto LABEL_28;
    }
    v3 = *(__int64 **)(a2 + 8);
    v4 = *(unsigned int *)(a2 + 20);
    v5 = &v3[v4];
    if ( v3 == v5 )
    {
LABEL_6:
      if ( (unsigned int)v4 < *(_DWORD *)(a2 + 16) )
      {
        *(_DWORD *)(a2 + 20) = v4 + 1;
        *v5 = a1;
        ++*(_QWORD *)a2;
        goto LABEL_8;
      }
      goto LABEL_26;
    }
    while ( a1 != *v3 )
    {
      if ( v5 == ++v3 )
        goto LABEL_6;
    }
  }
  return v2;
}
