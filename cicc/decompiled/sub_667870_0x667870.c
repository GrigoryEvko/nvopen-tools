// Function: sub_667870
// Address: 0x667870
//
__int64 __fastcall sub_667870(__int64 *a1)
{
  __int64 v1; // r12
  __int64 result; // rax
  __int64 v3; // rdx
  char v4; // cl
  char v5; // dl
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 i; // r12

  v1 = *a1;
  if ( !*a1 )
    return sub_6851C0(2386, a1 + 14);
  result = *(unsigned __int8 *)(v1 + 80);
  if ( (unsigned __int8)(result - 4) <= 1u )
    return sub_6851C0(2386, a1 + 14);
  if ( (_BYTE)result == 3 )
  {
    if ( (unsigned int)sub_8D3A70(*(_QWORD *)(v1 + 88)) )
      return sub_6851C0(2386, a1 + 14);
    result = *(unsigned __int8 *)(v1 + 80);
    if ( (_BYTE)result == 3 )
      return sub_6851C0(2386, a1 + 14);
  }
  if ( (*(_BYTE *)(v1 + 81) & 0x20) != 0 )
    goto LABEL_5;
  v3 = a1[36];
  if ( v3 )
  {
    while ( 1 )
    {
      v4 = *(_BYTE *)(v3 + 140);
      if ( v4 != 12 )
        break;
      v3 = *(_QWORD *)(v3 + 160);
    }
    if ( !v4 )
    {
LABEL_5:
      result = (unsigned int)(result - 10);
      if ( (unsigned __int8)result <= 1u )
      {
        result = *(_QWORD *)(v1 + 88);
        *(_BYTE *)(result + 193) &= ~2u;
      }
      return result;
    }
  }
  if ( (_BYTE)result == 10 )
  {
    v6 = *(_QWORD *)(v1 + 88);
    result = *(unsigned __int8 *)(v6 + 174);
    if ( (_BYTE)result == 2 )
    {
      result = dword_4D04880;
      if ( !dword_4D04880 )
      {
        result = *(_BYTE *)(v6 + 195) & 0xB;
        if ( (_BYTE)result != 1 )
          result = sub_6851C0(2395, a1 + 14);
        *(_BYTE *)(v6 + 193) &= ~2u;
      }
      return result;
    }
    if ( (_BYTE)result != 1 )
      return result;
LABEL_23:
    result = *(_QWORD *)(*(_QWORD *)(v6 + 40) + 32LL);
    if ( (*(_BYTE *)(result + 176) & 0x10) != 0 )
    {
      result = sub_6851C0(2403, a1 + 14);
      *(_BYTE *)(v6 + 193) &= ~2u;
    }
    return result;
  }
  if ( (_BYTE)result == 20 )
  {
    result = *(_QWORD *)(v1 + 88);
    v6 = *(_QWORD *)(result + 176);
    if ( *(_BYTE *)(v6 + 174) != 1 )
      return result;
    goto LABEL_23;
  }
  v5 = (result - 7) & 0xFD;
  if ( (_BYTE)result == 21 || !v5 )
  {
    v7 = *(_QWORD *)(v1 + 88);
    if ( (_BYTE)result != 9 && v5 )
      v7 = *(_QWORD *)(v7 + 192);
    result = sub_8D4130(*(_QWORD *)(v7 + 120));
    for ( i = result; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    if ( *(char *)(v7 + 171) < 0 )
    {
      result = sub_6851C0(2386, a1 + 14);
      *(_BYTE *)(v7 + 172) &= ~8u;
      return result;
    }
    if ( (*(_BYTE *)(i + 141) & 0x20) != 0 )
    {
      if ( (*((_BYTE *)a1 + 122) & 1) == 0 )
        return result;
    }
    else
    {
      result = sub_8D4160(i);
      if ( (_DWORD)result )
        return result;
      result = sub_8DBE70(i);
      if ( (_DWORD)result )
        return result;
      while ( 1 )
      {
        result = *(unsigned __int8 *)(i + 140);
        if ( (_BYTE)result != 12 )
          break;
        i = *(_QWORD *)(i + 160);
      }
      if ( !(_BYTE)result )
        return result;
      result = sub_6851C0(2402, a1 + 14);
    }
    *(_BYTE *)(v7 + 172) &= ~8u;
    return result;
  }
  if ( (_BYTE)result != 11 )
    return sub_6851C0(2386, a1 + 14);
  return result;
}
