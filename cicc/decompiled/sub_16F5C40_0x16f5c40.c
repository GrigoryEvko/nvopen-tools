// Function: sub_16F5C40
// Address: 0x16f5c40
//
const char *__fastcall sub_16F5C40(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // rdx
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r8
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // rcx
  _BYTE *v8; // rdx
  _BYTE *v9; // [rsp+0h] [rbp-20h] BYREF
  unsigned __int64 v10; // [rsp+8h] [rbp-18h]

  v9 = (_BYTE *)a1;
  v10 = a2;
  if ( a2 > 4 )
  {
    v2 = a1;
    if ( *(_DWORD *)a1 == 913142369 && *(_BYTE *)(a1 + 4) == 52 )
      goto LABEL_15;
    if ( *(_WORD *)a1 == 29281 )
    {
      v5 = 3;
      if ( *(_BYTE *)(a1 + 2) == 109 )
      {
LABEL_17:
        if ( *(_WORD *)(v2 + v5) == 25189 )
        {
          v6 = v10;
          v5 += 2LL;
          goto LABEL_19;
        }
LABEL_40:
        v6 = a2 - 2;
        if ( *(_WORD *)(v2 + a2 - 2) != 25189 )
          v6 = v10;
LABEL_19:
        if ( v6 >= v5 )
        {
          v7 = v6 - v5;
          v8 = (_BYTE *)(v5 + v2);
          v9 = v8;
          v10 = v7;
          if ( v7 )
          {
            if ( (v7 == 1 || *v8 == 118 && (unsigned int)((char)v8[1] - 48) <= 9)
              && sub_16D20C0((__int64 *)&v9, "eb", 2u, 0) == -1 )
            {
              return v9;
            }
            return byte_3F871B3;
          }
        }
        return (const char *)a1;
      }
    }
    if ( *(_DWORD *)a1 == 1836410996 && *(_BYTE *)(a1 + 4) == 98 )
    {
LABEL_15:
      v4 = a2 - 5;
      v5 = 5;
      goto LABEL_16;
    }
    if ( a2 > 6 && *(_DWORD *)a1 == 1668440417 && *(_WORD *)(a1 + 4) == 13928 && *(_BYTE *)(a1 + 6) == 52 )
    {
      if ( sub_16D20C0((__int64 *)&v9, "eb", 2u, 0) != -1 )
        return byte_3F871B3;
      a2 = v10;
      if ( v10 <= 6 )
      {
        if ( v10 <= 1 )
          return (const char *)a1;
        v2 = (__int64)v9;
        v5 = 7;
        goto LABEL_40;
      }
      v4 = v10 - 7;
      v2 = (__int64)v9;
      if ( v10 - 7 <= 2 )
      {
        v5 = 7;
      }
      else if ( *(_WORD *)(v9 + 7) != 25183 || (v4 = v10 - 10, v5 = 10, v9[9] != 101) )
      {
        v5 = 7;
        goto LABEL_17;
      }
LABEL_16:
      if ( v4 <= 1 )
        goto LABEL_40;
      goto LABEL_17;
    }
    goto LABEL_7;
  }
  if ( a2 > 2 )
  {
    v2 = a1;
    if ( *(_WORD *)a1 == 29281 )
    {
      v5 = 3;
      if ( *(_BYTE *)(a1 + 2) == 109 )
        goto LABEL_40;
    }
    goto LABEL_25;
  }
  if ( a2 == 2 )
  {
LABEL_25:
    v2 = (__int64)v9;
LABEL_7:
    if ( *(_WORD *)(v2 + a2 - 2) == 25189 )
    {
      v10 = a2 - 2;
      if ( a2 == 2 )
        return (const char *)a1;
    }
    return v9;
  }
  if ( !a2 )
    return (const char *)a1;
  return v9;
}
