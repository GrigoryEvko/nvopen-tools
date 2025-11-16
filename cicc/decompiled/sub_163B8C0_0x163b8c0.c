// Function: sub_163B8C0
// Address: 0x163b8c0
//
char __fastcall sub_163B8C0(__int64 a1)
{
  char v1; // al
  char result; // al
  __int64 v3; // rdx
  _QWORD *v4; // rbx
  __int64 v5; // r12
  _QWORD *v6; // r13
  __int64 v7; // rax
  __int64 v8; // r12
  _QWORD *v9; // r12

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 8);
    if ( v1 == 15 )
      return *(_DWORD *)(a1 + 8) >> 8 == 1;
    if ( v1 == 16 )
    {
      v3 = **(_QWORD **)(a1 + 16);
      result = 0;
      if ( *(_BYTE *)(v3 + 8) == 15 )
        return *(_DWORD *)(v3 + 8) >> 8 == 1;
      return result;
    }
    if ( v1 != 14 )
      break;
    a1 = *(_QWORD *)(a1 + 24);
  }
  if ( v1 != 13 )
    return 0;
  v4 = *(_QWORD **)(a1 + 16);
  v5 = 8LL * *(unsigned int *)(a1 + 12);
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( !v8 )
  {
LABEL_21:
    if ( v7 != 2 )
    {
      if ( v7 != 3 )
      {
        if ( v7 != 1 )
          return 0;
        goto LABEL_33;
      }
      if ( (unsigned __int8)sub_163B8C0(*v4) )
        return v6 != v4;
      ++v4;
    }
    if ( (unsigned __int8)sub_163B8C0(*v4) )
      return v6 != v4;
    ++v4;
LABEL_33:
    result = sub_163B8C0(*v4);
    if ( result )
      return v6 != v4;
    return result;
  }
  v9 = &v4[4 * v8];
  while ( !(unsigned __int8)sub_163B8C0(*v4) )
  {
    if ( (unsigned __int8)sub_163B8C0(v4[1]) )
      return v6 != v4 + 1;
    if ( (unsigned __int8)sub_163B8C0(v4[2]) )
      return v6 != v4 + 2;
    if ( (unsigned __int8)sub_163B8C0(v4[3]) )
      return v6 != v4 + 3;
    v4 += 4;
    if ( v4 == v9 )
    {
      v7 = v6 - v4;
      goto LABEL_21;
    }
  }
  return v6 != v4;
}
