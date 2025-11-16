// Function: sub_666F40
// Address: 0x666f40
//
__int64 __fastcall sub_666F40(__int64 *a1)
{
  __int64 v1; // rcx
  __int64 v2; // rax
  char v3; // dl
  __int64 result; // rax
  __int64 v5; // r12

  v1 = *a1;
  if ( !*a1 )
    return sub_6851C0(2926, a1 + 14);
  if ( (*(_BYTE *)(v1 + 81) & 0x20) == 0 )
  {
    v2 = a1[36];
    if ( !v2 )
      goto LABEL_11;
    while ( 1 )
    {
      v3 = *(_BYTE *)(v2 + 140);
      if ( v3 != 12 )
        break;
      v2 = *(_QWORD *)(v2 + 160);
    }
    if ( v3 )
    {
LABEL_11:
      result = *(unsigned __int8 *)(v1 + 80);
      if ( (_BYTE)result == 10 )
      {
        v5 = *(_QWORD *)(v1 + 88);
        result = *(unsigned __int8 *)(v5 + 174);
        if ( (_BYTE)result == 2 )
        {
          result = *(_BYTE *)(v5 + 195) & 0xB;
          if ( (_BYTE)result != 1 )
            result = sub_6851C0(2927, a1 + 14);
          *(_BYTE *)(v5 + 193) &= 0xF9u;
        }
        else if ( (_BYTE)result == 1 )
        {
          goto LABEL_23;
        }
      }
      else
      {
        if ( (_BYTE)result != 20 )
        {
          if ( (((_BYTE)result - 7) & 0xFD) == 0 || (_BYTE)result == 21 )
            return sub_6851C0(2929, a1 + 14);
          if ( (_BYTE)result == 11 )
          {
            v5 = *(_QWORD *)(v1 + 88);
            if ( !v5 )
              return result;
            goto LABEL_17;
          }
          return sub_6851C0(2926, a1 + 14);
        }
        result = *(_QWORD *)(v1 + 88);
        v5 = *(_QWORD *)(result + 176);
        if ( *(_BYTE *)(v5 + 174) == 1 )
        {
LABEL_23:
          result = *(_QWORD *)(*(_QWORD *)(v5 + 40) + 32LL);
          if ( (*(_BYTE *)(result + 176) & 0x10) == 0 )
            return result;
          result = sub_6851C0(2928, a1 + 14);
          *(_BYTE *)(v5 + 193) &= 0xF9u;
        }
      }
LABEL_17:
      if ( (*(_BYTE *)(v5 + 193) & 4) != 0 && *(_BYTE *)(v5 + 174) == 5 )
      {
        result = (unsigned int)*(unsigned __int8 *)(v5 + 176) - 1;
        if ( (unsigned __int8)(*(_BYTE *)(v5 + 176) - 1) <= 3u )
          return sub_6851C0(2959, a1 + 14);
      }
      return result;
    }
  }
  result = (unsigned int)*(unsigned __int8 *)(v1 + 80) - 10;
  if ( (unsigned __int8)(*(_BYTE *)(v1 + 80) - 10) <= 1u )
  {
    *(_BYTE *)(*(_QWORD *)(v1 + 88) + 193LL) &= ~2u;
    result = *(_QWORD *)(v1 + 88);
    *(_BYTE *)(result + 193) &= ~4u;
  }
  return result;
}
