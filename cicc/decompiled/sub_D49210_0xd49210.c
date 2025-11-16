// Function: sub_D49210
// Address: 0xd49210
//
__int64 __fastcall sub_D49210(__int64 a1)
{
  __int64 v1; // r13
  __int64 v2; // r14
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rax
  _QWORD *i; // r15
  __int64 v7; // [rsp+8h] [rbp-38h]

  v1 = 0x8000000000041LL;
  v2 = *(_QWORD *)(a1 + 32);
  v7 = *(_QWORD *)(a1 + 40);
  if ( v7 == v2 )
    return 1;
  while ( 1 )
  {
    v3 = (_QWORD *)(*(_QWORD *)v2 + 48LL);
    v4 = *v3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 == (_QWORD *)v4 )
      goto LABEL_17;
    if ( !v4 )
      BUG();
    if ( (unsigned int)*(unsigned __int8 *)(v4 - 24) - 30 > 0xA )
LABEL_17:
      BUG();
    if ( *(_BYTE *)(v4 - 24) == 33 )
      return 0;
    for ( i = *(_QWORD **)(*(_QWORD *)v2 + 56LL); i != v3; i = (_QWORD *)i[1] )
    {
      if ( !i )
        BUG();
      if ( (unsigned __int8)(*((_BYTE *)i - 24) - 34) <= 0x33u
        && _bittest64(&v1, (unsigned int)*((unsigned __int8 *)i - 24) - 34)
        && ((unsigned __int8)sub_A73ED0(i + 6, 27) || (unsigned __int8)sub_B49560((__int64)(i - 3), 27)) )
      {
        return 0;
      }
    }
    v2 += 8;
    if ( v7 == v2 )
      return 1;
  }
}
