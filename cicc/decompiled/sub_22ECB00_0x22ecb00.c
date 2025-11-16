// Function: sub_22ECB00
// Address: 0x22ecb00
//
char __fastcall sub_22ECB00(__int64 a1)
{
  unsigned __int8 v1; // dl
  __int64 v2; // rdx
  char result; // al
  __int64 v4; // rax
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  _QWORD *v7; // r13

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 8);
    if ( v1 == 14 && *(_DWORD *)(a1 + 8) >> 8 == 1 )
      return 1;
    if ( (unsigned int)v1 - 17 <= 1 )
    {
      v2 = **(_QWORD **)(a1 + 16);
      result = 0;
      if ( *(_BYTE *)(v2 + 8) == 14 )
        return *(_DWORD *)(v2 + 8) >> 8 == 1;
      return result;
    }
    if ( v1 != 16 )
      break;
    a1 = *(_QWORD *)(a1 + 24);
  }
  result = 0;
  if ( v1 != 15 )
    return result;
  v4 = *(unsigned int *)(a1 + 12);
  v5 = *(_QWORD **)(a1 + 16);
  v6 = &v5[v4];
  if ( (8 * v4) >> 5 )
  {
    v7 = &v5[4 * ((8 * v4) >> 5)];
    while ( !(unsigned __int8)sub_22ECB00(*v5) )
    {
      if ( (unsigned __int8)sub_22ECB00(v5[1]) )
        return v6 != v5 + 1;
      if ( (unsigned __int8)sub_22ECB00(v5[2]) )
        return v6 != v5 + 2;
      if ( (unsigned __int8)sub_22ECB00(v5[3]) )
        return v6 != v5 + 3;
      v5 += 4;
      if ( v7 == v5 )
      {
        v4 = v6 - v5;
        goto LABEL_20;
      }
    }
    return v6 != v5;
  }
LABEL_20:
  if ( v4 == 2 )
  {
LABEL_30:
    if ( !(unsigned __int8)sub_22ECB00(*v5) )
    {
      ++v5;
      goto LABEL_32;
    }
    return v6 != v5;
  }
  if ( v4 == 3 )
  {
    if ( (unsigned __int8)sub_22ECB00(*v5) )
      return v6 != v5;
    ++v5;
    goto LABEL_30;
  }
  if ( v4 != 1 )
    return 0;
LABEL_32:
  result = sub_22ECB00(*v5);
  if ( result )
    return v6 != v5;
  return result;
}
