// Function: sub_154A7B0
// Address: 0x154a7b0
//
unsigned __int64 __fastcall sub_154A7B0(__int64 a1, __int64 a2)
{
  unsigned __int64 result; // rax
  char v4; // dl
  char v5; // al
  const char *v6; // rsi
  char v7; // al
  __int64 v8; // rdx
  char v9; // dl
  void *v10; // rdx

  result = *(unsigned __int8 *)(a2 + 16);
  if ( (unsigned __int8)result <= 0x17u )
  {
    if ( (_BYTE)result != 5 )
      return result;
    v7 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
    if ( v7 == 16 )
      v7 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
    if ( (unsigned __int8)(v7 - 1) > 5u )
    {
      result = *(unsigned __int16 *)(a2 + 18);
      if ( (_WORD)result != 52 )
        goto LABEL_32;
    }
LABEL_6:
    v5 = *(_BYTE *)(a2 + 17) >> 1;
    if ( (v5 & 0x3F) == 0x3F && (v5 & 0x40) != 0 )
    {
      sub_1263B40(a1, " fast");
      goto LABEL_14;
    }
    if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
    {
      sub_1263B40(a1, " reassoc");
      v5 = *(_BYTE *)(a2 + 17) >> 1;
      if ( (v5 & 2) == 0 )
      {
LABEL_9:
        if ( (v5 & 4) == 0 )
          goto LABEL_10;
        goto LABEL_43;
      }
    }
    else if ( (v5 & 2) == 0 )
    {
      goto LABEL_9;
    }
    sub_1263B40(a1, " nnan");
    v5 = *(_BYTE *)(a2 + 17) >> 1;
    if ( (v5 & 4) == 0 )
    {
LABEL_10:
      if ( (v5 & 8) == 0 )
        goto LABEL_11;
      goto LABEL_44;
    }
LABEL_43:
    sub_1263B40(a1, " ninf");
    v5 = *(_BYTE *)(a2 + 17) >> 1;
    if ( (v5 & 8) == 0 )
    {
LABEL_11:
      if ( (v5 & 0x10) == 0 )
        goto LABEL_12;
      goto LABEL_45;
    }
LABEL_44:
    sub_1263B40(a1, " nsz");
    v5 = *(_BYTE *)(a2 + 17) >> 1;
    if ( (v5 & 0x10) == 0 )
    {
LABEL_12:
      if ( (v5 & 0x20) == 0 )
        goto LABEL_13;
      goto LABEL_46;
    }
LABEL_45:
    sub_1263B40(a1, " arcp");
    v5 = *(_BYTE *)(a2 + 17) >> 1;
    if ( (v5 & 0x20) == 0 )
    {
LABEL_13:
      if ( (v5 & 0x40) == 0 )
      {
LABEL_14:
        result = *(unsigned __int8 *)(a2 + 16);
        if ( (unsigned __int8)result > 0x17u )
          goto LABEL_15;
        if ( (_BYTE)result != 5 )
          return result;
        result = *(unsigned __int16 *)(a2 + 18);
LABEL_32:
        if ( (unsigned __int16)result > 0x17u )
        {
          if ( (unsigned __int16)(result - 24) > 1u )
          {
            if ( (_WORD)result != 32 )
              return result;
            goto LABEL_37;
          }
        }
        else
        {
          v10 = &loc_80A800;
          if ( _bittest64((const __int64 *)&v10, result) )
          {
LABEL_26:
            result = *(unsigned __int8 *)(a2 + 17);
            v9 = *(_BYTE *)(a2 + 17) >> 1;
            if ( (result & 2) != 0 )
            {
              result = sub_1263B40(a1, " nuw");
              v9 = *(_BYTE *)(a2 + 17) >> 1;
            }
            v6 = " nsw";
            if ( (v9 & 2) != 0 )
              return sub_1263B40(a1, v6);
            return result;
          }
          if ( (unsigned int)(unsigned __int16)result - 17 > 1 )
            return result;
        }
LABEL_17:
        v6 = " exact";
        if ( (*(_BYTE *)(a2 + 17) & 2) == 0 )
          return result;
        return sub_1263B40(a1, v6);
      }
LABEL_47:
      sub_1263B40(a1, " afn");
      goto LABEL_14;
    }
LABEL_46:
    sub_1263B40(a1, " contract");
    if ( ((*(_BYTE *)(a2 + 17) >> 1) & 0x40) == 0 )
      goto LABEL_14;
    goto LABEL_47;
  }
  v4 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  if ( v4 == 16 )
    v4 = *(_BYTE *)(**(_QWORD **)(*(_QWORD *)a2 + 16LL) + 8LL);
  if ( (unsigned __int8)(v4 - 1) <= 5u || (_BYTE)result == 76 )
    goto LABEL_6;
LABEL_15:
  if ( (unsigned __int8)result <= 0x2Fu )
  {
    v8 = 0x80A800000000LL;
    if ( _bittest64(&v8, result) )
      goto LABEL_26;
    if ( (unsigned int)(unsigned __int8)result - 41 > 1 )
      return result;
    goto LABEL_17;
  }
  if ( (unsigned __int8)(result - 48) <= 1u )
    goto LABEL_17;
  if ( (_BYTE)result == 56 )
  {
LABEL_37:
    if ( (*(_BYTE *)(a2 + 17) & 2) != 0 )
    {
      v6 = " inbounds";
      return sub_1263B40(a1, v6);
    }
  }
  return result;
}
