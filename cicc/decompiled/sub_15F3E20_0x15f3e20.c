// Function: sub_15F3E20
// Address: 0x15f3e20
//
char __fastcall sub_15F3E20(__int64 a1, __int64 a2, char a3)
{
  char v4; // dl
  char result; // al
  unsigned __int16 v6; // cx
  unsigned int v7; // edx
  int v8; // edx
  int v9; // eax
  __int64 v10; // rdx
  size_t v11; // rdx
  unsigned int v12; // edx
  unsigned int v13; // ecx
  unsigned int v14; // edx
  unsigned int v15; // ecx

  v4 = *(_BYTE *)(a1 + 16);
  if ( v4 == 53 )
  {
    result = 0;
    if ( *(_QWORD *)(a1 + 56) == *(_QWORD *)(a2 + 56) )
      return a3 | ((unsigned int)(1 << *(_WORD *)(a2 + 18)) >> 1 == (unsigned int)(1 << *(_WORD *)(a1 + 18)) >> 1);
    return result;
  }
  if ( v4 == 54 || v4 == 55 )
  {
    v6 = *(_WORD *)(a1 + 18);
    v7 = *(unsigned __int16 *)(a2 + 18);
    result = 0;
    if ( (v6 & 1) == (*(_WORD *)(a2 + 18) & 1) )
    {
      result = a3 | (1 << (v7 >> 1) >> 1 == 1 << (v6 >> 1) >> 1);
      if ( result )
      {
        result = 0;
        if ( ((v7 >> 7) & 7) == ((v6 >> 7) & 7) )
          return *(_BYTE *)(a1 + 56) == *(_BYTE *)(a2 + 56);
      }
    }
  }
  else
  {
    if ( (unsigned __int8)(v4 - 75) <= 1u )
    {
      v8 = *(unsigned __int16 *)(a2 + 18);
      v9 = *(unsigned __int16 *)(a1 + 18);
      BYTE1(v8) &= ~0x80u;
      BYTE1(v9) &= ~0x80u;
      return v8 == v9;
    }
    if ( v4 == 78 )
    {
      result = 0;
      if ( (*(_WORD *)(a2 + 18) & 3u) - 1 <= 1 == (*(_WORD *)(a1 + 18) & 3u) - 1 <= 1
        && (unsigned int)(*(unsigned __int16 *)(a2 + 18) << 17) >> 19 == (unsigned int)(*(unsigned __int16 *)(a1 + 18) << 17) >> 19
        && *(_QWORD *)(a2 + 56) == *(_QWORD *)(a1 + 56) )
      {
        return sub_15F3C00(a1, a2);
      }
    }
    else if ( v4 == 29 )
    {
      result = 0;
      if ( ((*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF) == ((*(unsigned __int16 *)(a1 + 18) >> 2) & 0x3FFFDFFF)
        && *(_QWORD *)(a1 + 56) == *(_QWORD *)(a2 + 56) )
      {
        return sub_15F3D10(a1, a2);
      }
    }
    else
    {
      if ( v4 != 87 && v4 != 86 )
      {
        if ( v4 == 57 )
        {
          result = 0;
          if ( ((*(unsigned __int16 *)(a2 + 18) >> 1) & 0x7FFFBFFF) != ((*(unsigned __int16 *)(a1 + 18) >> 1)
                                                                      & 0x7FFFBFFF) )
            return result;
        }
        else if ( v4 == 58 )
        {
          v12 = *(unsigned __int16 *)(a1 + 18);
          v13 = *(unsigned __int16 *)(a2 + 18);
          result = 0;
          if ( (*(_WORD *)(a2 + 18) & 1) != (*(_WORD *)(a1 + 18) & 1)
            || (BYTE1(v13) & 1) != (BYTE1(v12) & 1)
            || ((v13 >> 2) & 7) != ((v12 >> 2) & 7)
            || (unsigned __int8)v13 >> 5 != (unsigned __int8)v12 >> 5 )
          {
            return result;
          }
        }
        else
        {
          result = 1;
          if ( v4 != 59 )
            return result;
          v14 = *(unsigned __int16 *)(a1 + 18);
          v15 = *(unsigned __int16 *)(a2 + 18);
          result = 0;
          if ( ((v15 >> 5) & 0x7FFFBFF) != ((v14 >> 5) & 0x7FFFBFF)
            || (*(_WORD *)(a2 + 18) & 1) != (*(_WORD *)(a1 + 18) & 1)
            || ((v15 >> 2) & 7) != ((v14 >> 2) & 7) )
          {
            return result;
          }
        }
        return *(_BYTE *)(a2 + 56) == *(_BYTE *)(a1 + 56);
      }
      v10 = *(unsigned int *)(a1 + 64);
      result = 0;
      if ( v10 == *(_DWORD *)(a2 + 64) )
      {
        v11 = 4 * v10;
        result = 1;
        if ( v11 )
          return memcmp(*(const void **)(a1 + 56), *(const void **)(a2 + 56), v11) == 0;
      }
    }
  }
  return result;
}
