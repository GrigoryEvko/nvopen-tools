// Function: sub_8DB8B0
// Address: 0x8db8b0
//
__int64 __fastcall sub_8DB8B0(_BYTE *a1, _DWORD *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdx
  char v6; // al
  __int64 result; // rax
  int v8; // eax
  __int64 v9; // rcx
  unsigned __int64 v10; // rax

  v5 = (unsigned int)dword_4F60568;
  if ( dword_4F60568 )
  {
    v6 = a1[142];
    if ( (v6 & 2) != 0 )
    {
      *a2 = 1;
      return v6 & 1;
    }
  }
  v8 = (unsigned __int8)a1[140];
  v9 = (unsigned int)(v8 - 9);
  if ( (unsigned __int8)(v8 - 9) > 2u )
  {
    if ( (_BYTE)v8 == 14 )
    {
      if ( qword_4F60580 && (_BYTE *)qword_4F60580 != a1 )
      {
        if ( !(unsigned int)sub_8D97D0((__int64)a1, qword_4F60580, 0, v9, a5) )
          return 0;
        LODWORD(v5) = dword_4F60568;
      }
    }
    else
    {
      v9 = (unsigned int)dword_4F6056C;
      if ( !dword_4F6056C )
        goto LABEL_19;
      if ( (_BYTE)v8 == 2 )
      {
        if ( (a1[161] & 8) == 0 || (a1[162] & 0x40) == 0 )
          goto LABEL_19;
      }
      else if ( (_BYTE)v8 == 12 )
      {
        if ( (a1[186] & 0x18) == 0 )
        {
          v10 = (unsigned __int8)a1[184];
          if ( (unsigned __int8)v10 <= 0xCu )
          {
            v5 = 6338;
            if ( _bittest64(&v5, v10) )
            {
              *a2 = 1;
              return 0;
            }
          }
          goto LABEL_19;
        }
      }
      else if ( (_BYTE)v8 != 8 || (char)a1[168] >= 0 )
      {
        goto LABEL_19;
      }
    }
LABEL_8:
    *a2 = 1;
    result = 1;
    if ( (_DWORD)v5 )
      goto LABEL_9;
    return result;
  }
  if ( (a1[177] & 0x20) == 0 )
    return 0;
  if ( dword_4F6056C )
    goto LABEL_8;
LABEL_19:
  if ( qword_4F60580 )
    return 0;
  result = sub_8D1700((__int64)a1, a2, v5, v9);
  if ( (_DWORD)result )
  {
    if ( dword_4F60568 )
      goto LABEL_9;
    return result;
  }
  if ( (unsigned __int8)(a1[140] - 9) > 2u || (a1[177] & 0x20) == 0 )
    return 0;
  result = sub_8D1BE0((__int64)a1, a2);
  if ( dword_4F60568 )
  {
    if ( !(_DWORD)result )
      return 0;
LABEL_9:
    a1[142] |= 3u;
  }
  return result;
}
