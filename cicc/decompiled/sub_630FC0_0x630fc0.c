// Function: sub_630FC0
// Address: 0x630fc0
//
__int64 __fastcall sub_630FC0(__int64 a1)
{
  __int64 v1; // r12
  char v2; // al
  unsigned int v3; // r13d
  __int64 *v4; // rax
  __int64 v5; // rbx
  __int64 v6; // rax
  __int64 v7; // r13
  __int64 **v8; // rbx
  __int64 *v9; // r12

  v1 = a1;
  if ( (unsigned int)sub_8D3410(a1) )
    v1 = sub_8D40F0(a1);
  while ( 1 )
  {
    v2 = *(_BYTE *)(v1 + 140);
    if ( v2 != 12 )
      break;
    v1 = *(_QWORD *)(v1 + 160);
  }
  v3 = 0;
  if ( (unsigned __int8)(v2 - 9) > 2u )
    return v3;
  v4 = *(__int64 **)(*(_QWORD *)v1 + 96LL);
  if ( (*((_BYTE *)v4 + 178) & 0x40) == 0 )
    return v3;
  v5 = *v4;
  if ( !*v4 )
  {
LABEL_22:
    v3 = 0;
    if ( !unk_4D04418 )
      return v3;
    goto LABEL_15;
  }
  while ( 1 )
  {
    if ( *(_BYTE *)(v5 + 80) == 8 )
    {
      v6 = *(_QWORD *)(v5 + 88);
      if ( (*(_BYTE *)(v6 + 145) & 0x20) == 0 )
      {
        v7 = *(_QWORD *)(v6 + 120);
        if ( (unsigned int)sub_8D32E0(v7) || (unsigned int)sub_630FC0(v7) )
          break;
      }
    }
    if ( *(_BYTE *)(v1 + 140) != 11 )
    {
      v5 = *(_QWORD *)(v5 + 16);
      if ( v5 )
        continue;
    }
    goto LABEL_22;
  }
  v3 = 1;
  if ( unk_4D04418 )
  {
LABEL_15:
    v8 = **(__int64 ****)(v1 + 168);
    if ( v8 )
    {
      while ( 1 )
      {
        if ( ((_BYTE)v8[12] & 1) != 0 )
        {
          v9 = v8[5];
          if ( (unsigned int)sub_8D32E0(v9) || (unsigned int)sub_630FC0(v9) )
            break;
        }
        v8 = (__int64 **)*v8;
        if ( !v8 )
          return v3;
      }
      return 1;
    }
  }
  return v3;
}
