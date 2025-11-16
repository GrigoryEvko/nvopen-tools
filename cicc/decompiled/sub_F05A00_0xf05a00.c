// Function: sub_F05A00
// Address: 0xf05a00
//
const char *__fastcall sub_F05A00(_WORD *s1, unsigned __int64 a2)
{
  _WORD *v2; // r14
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // r12
  _BYTE *v6; // rdx
  _WORD *v8; // [rsp+0h] [rbp-30h] BYREF
  unsigned __int64 v9; // [rsp+8h] [rbp-28h]

  v2 = s1;
  v3 = a2;
  v8 = s1;
  v9 = a2;
  if ( a2 > 7 )
  {
    v4 = 8;
    if ( *(_QWORD *)s1 == 0x32335F34366D7261LL )
      goto LABEL_3;
LABEL_19:
    if ( *(_DWORD *)s1 == 913142369 )
    {
      v4 = 6;
      if ( s1[2] == 25908 )
        goto LABEL_28;
    }
    if ( *(_DWORD *)s1 == 913142369 && *((_BYTE *)s1 + 4) == 52 )
    {
LABEL_27:
      v4 = 5;
      goto LABEL_28;
    }
    if ( a2 > 9 )
    {
      if ( *(_QWORD *)s1 == 0x5F34366863726161LL )
      {
        v4 = 10;
        if ( s1[4] == 12851 )
        {
LABEL_3:
          if ( a2 - v4 <= 1 )
            goto LABEL_31;
          goto LABEL_4;
        }
      }
      if ( *s1 == 29281 && *((_BYTE *)s1 + 2) == 109 )
        goto LABEL_53;
      if ( *(_DWORD *)s1 == 1836410996 && *((_BYTE *)s1 + 4) == 98 )
      {
LABEL_26:
        v2 = v8;
        v3 = v9;
        goto LABEL_27;
      }
LABEL_39:
      if ( *(_DWORD *)s1 == 1668440417 && s1[2] == 13928 && *((_BYTE *)s1 + 6) == 52 )
      {
        if ( sub_C931B0((__int64 *)&v8, "eb", 2u, 0) != -1 )
          return byte_3F871B3;
        v3 = v9;
        v2 = v8;
        v4 = 7;
        if ( v9 > 9 )
        {
          if ( *(_WORD *)((char *)v8 + 7) != 25183 || (v4 = 10, *((_BYTE *)v8 + 9) != 101) )
            v4 = 7;
        }
LABEL_28:
        if ( v4 > v3 || v3 - v4 <= 1 )
        {
          if ( v3 <= 1 )
            return (const char *)s1;
          goto LABEL_31;
        }
LABEL_4:
        if ( *(_WORD *)((char *)v2 + v4) == 25189 )
        {
          v4 += 2LL;
          goto LABEL_6;
        }
LABEL_31:
        if ( *(_WORD *)((char *)v2 + v3 - 2) == 25189 )
          v3 -= 2LL;
LABEL_6:
        if ( v3 >= v4 )
        {
          v5 = v3 - v4;
          v6 = (char *)v2 + v4;
          v8 = v6;
          v9 = v5;
          if ( v5 )
          {
            if ( (v5 == 1 || *v6 == 118 && (unsigned int)((char)v6[1] - 48) <= 9)
              && sub_C931B0((__int64 *)&v8, "eb", 2u, 0) == -1 )
            {
              return (const char *)v8;
            }
            return byte_3F871B3;
          }
        }
        return (const char *)s1;
      }
LABEL_40:
      v2 = v8;
      v3 = v9;
      goto LABEL_14;
    }
    goto LABEL_13;
  }
  if ( a2 > 5 )
    goto LABEL_19;
  if ( a2 == 5 )
  {
    if ( *(_DWORD *)s1 == 913142369 && *((_BYTE *)s1 + 4) == 52 )
      goto LABEL_27;
    goto LABEL_35;
  }
LABEL_13:
  if ( a2 <= 2 )
    goto LABEL_14;
LABEL_35:
  if ( !memcmp(s1, &unk_3F8856D, 3u) )
  {
LABEL_53:
    v4 = 3;
    goto LABEL_28;
  }
  if ( a2 > 4 )
  {
    if ( !memcmp(s1, "thumb", 5u) )
      goto LABEL_26;
    if ( a2 <= 6 )
      goto LABEL_40;
    goto LABEL_39;
  }
LABEL_14:
  if ( v3 > 1 )
  {
    v3 -= 2LL;
    if ( *(_WORD *)((char *)v2 + v3) != 25189 )
      return (const char *)v8;
    v9 = v3;
  }
  if ( !v3 )
    return (const char *)s1;
  return (const char *)v8;
}
