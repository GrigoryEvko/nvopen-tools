// Function: sub_B2E3E0
// Address: 0xb2e3e0
//
__int64 __fastcall sub_B2E3E0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 v5; // r12
  __int64 v6; // rax

  v1 = a1 + 72;
  v2 = *(_QWORD *)(a1 + 80);
  if ( a1 + 72 == v2 )
  {
    i = 0;
  }
  else
  {
    if ( !v2 )
      BUG();
    while ( 1 )
    {
      i = *(_QWORD *)(v2 + 32);
      if ( i != v2 + 24 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        return 0;
      if ( !v2 )
        BUG();
    }
  }
  v5 = 0x8000000000041LL;
  if ( v1 == v2 )
    return 0;
  while ( 1 )
  {
    if ( !i )
      BUG();
    if ( (unsigned __int8)(*(_BYTE *)(i - 24) - 34) <= 0x33u
      && _bittest64(&v5, (unsigned int)*(unsigned __int8 *)(i - 24) - 34)
      && ((unsigned __int8)sub_A73ED0((_QWORD *)(i + 48), 53) || (unsigned __int8)sub_B49560(i - 24, 53)) )
    {
      break;
    }
    for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v2 + 32) )
    {
      v6 = v2 - 24;
      if ( !v2 )
        v6 = 0;
      if ( i != v6 + 48 )
        break;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v1 == v2 )
        return 0;
      if ( !v2 )
        BUG();
    }
    if ( v2 == v1 )
      return 0;
  }
  return 1;
}
