// Function: sub_994E00
// Address: 0x994e00
//
__int64 __fastcall sub_994E00(__int64 *a1, char *a2)
{
  unsigned __int8 v2; // al
  __int64 v3; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r12
  __int64 v10; // rcx
  __int64 v11; // r13
  __int16 v12; // ax
  int v13; // eax
  unsigned int v14; // eax
  __int64 v15; // rdx

  v2 = *a2;
  if ( (unsigned __int8)*a2 <= 0x1Cu )
    return 0;
  if ( v2 == 85 )
  {
    v5 = *((_QWORD *)a2 - 4);
    if ( v5
      && !*(_BYTE *)v5
      && *(_QWORD *)(v5 + 24) == *((_QWORD *)a2 + 10)
      && (*(_BYTE *)(v5 + 33) & 0x20) != 0
      && *(_DWORD *)(v5 + 36) == 329 )
    {
      v6 = *((_DWORD *)a2 + 1) & 0x7FFFFFF;
      v7 = 32 * (1 - v6);
      LOBYTE(v7) = *(_QWORD *)&a2[v7] == *a1;
      v6 *= -32;
      LOBYTE(v6) = *(_QWORD *)&a2[v6] == *a1;
      return (unsigned int)v6 | (unsigned int)v7;
    }
    return 0;
  }
  if ( v2 == 86 )
  {
    v3 = *((_QWORD *)a2 - 12);
    if ( *(_BYTE *)v3 == 82 )
    {
      v8 = *((_QWORD *)a2 - 8);
      v9 = *(_QWORD *)(v3 - 64);
      v10 = *((_QWORD *)a2 - 4);
      v11 = *(_QWORD *)(v3 - 32);
      if ( v9 == v8 && v11 == v10 )
      {
        v12 = *(_WORD *)(v3 + 2);
      }
      else
      {
        if ( v11 != v8 || v9 != v10 )
          return 0;
        v12 = *(_WORD *)(v3 + 2);
        if ( v9 != v8 )
        {
          v13 = sub_B52870(v12 & 0x3F);
LABEL_18:
          v14 = v13 - 38;
          if ( v14 <= 1 )
          {
            v15 = *a1;
            LOBYTE(v14) = v9 == *a1;
            LOBYTE(v15) = v11 == *a1;
            return (unsigned int)v15 | v14;
          }
          return 0;
        }
      }
      v13 = v12 & 0x3F;
      goto LABEL_18;
    }
  }
  return 0;
}
