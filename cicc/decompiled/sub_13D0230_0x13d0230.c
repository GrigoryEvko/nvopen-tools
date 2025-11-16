// Function: sub_13D0230
// Address: 0x13d0230
//
__int64 __fastcall sub_13D0230(_QWORD *a1, __int64 a2, char a3, char a4)
{
  unsigned __int8 v5; // al
  __int64 v6; // r8
  unsigned int v8; // esi
  __int64 v9; // rax
  int v10; // edi
  __int64 v11; // rax
  __int64 *v12; // rdx
  int v13; // r12d
  unsigned int v14; // r13d
  __int64 v15; // rax
  char v16; // dl

  v5 = *((_BYTE *)a1 + 16);
  if ( v5 != 9 )
  {
    if ( v5 <= 0x17u )
    {
      if ( v5 != 5 )
      {
        if ( !a4 )
          return 0;
        if ( v5 == 13 )
        {
          v8 = *((_DWORD *)a1 + 8);
          v9 = a1[3];
          if ( v8 > 0x40 )
            v9 = *(_QWORD *)(v9 + 8LL * ((v8 - 1) >> 6));
          if ( (v9 & (1LL << ((unsigned __int8)v8 - 1))) != 0 )
            return (__int64)a1;
          return 0;
        }
        goto LABEL_27;
      }
      v10 = *((unsigned __int16 *)a1 + 9);
      if ( ((unsigned int)(v10 - 17) <= 1 || (unsigned __int16)(v10 - 24) <= 1u)
        && (*((_BYTE *)a1 + 17) & 2) != 0
        && (unsigned int)(v10 - 24) <= 1 )
      {
        v6 = a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
        if ( v6 )
        {
          if ( a2 == a1[3 * (1LL - (*((_DWORD *)a1 + 5) & 0xFFFFFFF))] )
            return v6;
        }
      }
    }
    else if ( ((unsigned __int8)(v5 - 48) <= 1u || (unsigned int)v5 - 41 <= 1)
           && (*((_BYTE *)a1 + 17) & 2) != 0
           && (unsigned int)v5 - 48 <= 1 )
    {
      if ( (*((_BYTE *)a1 + 23) & 0x40) != 0 )
      {
        v12 = (__int64 *)*(a1 - 1);
        v6 = *v12;
        if ( !*v12 )
          goto LABEL_10;
      }
      else
      {
        v12 = &a1[-3 * (*((_DWORD *)a1 + 5) & 0xFFFFFFF)];
        v6 = *v12;
        if ( !*v12 )
          goto LABEL_10;
      }
      if ( a2 == v12[3] )
        return v6;
    }
LABEL_10:
    if ( !a4 )
      return 0;
LABEL_27:
    if ( *(_BYTE *)(*a1 + 8LL) != 16 || v5 > 0x10u )
      return 0;
    v11 = sub_15A1020(a1);
    if ( v11 && *(_BYTE *)(v11 + 16) == 13 )
    {
      if ( !sub_13D0200((__int64 *)(v11 + 24), *(_DWORD *)(v11 + 32) - 1) )
        return 0;
    }
    else
    {
      v13 = *(_QWORD *)(*a1 + 32LL);
      if ( v13 )
      {
        v14 = 0;
        while ( 1 )
        {
          v15 = sub_15A0A60(a1, v14);
          if ( !v15 )
            break;
          v16 = *(_BYTE *)(v15 + 16);
          if ( v16 != 9 && (v16 != 13 || !sub_13D0200((__int64 *)(v15 + 24), *(_DWORD *)(v15 + 32) - 1)) )
            break;
          if ( v13 == ++v14 )
            return (__int64)a1;
        }
        return 0;
      }
    }
    return (__int64)a1;
  }
  if ( a4 || a3 )
    return (__int64)a1;
  return sub_15A06D0(*a1);
}
