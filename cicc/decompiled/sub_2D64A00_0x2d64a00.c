// Function: sub_2D64A00
// Address: 0x2d64a00
//
__int64 __fastcall sub_2D64A00(__int64 a1)
{
  _QWORD *v1; // rdx
  _QWORD *v2; // r8
  _QWORD *i; // rax
  char v4; // cl
  __int64 v5; // r12
  __int64 v7; // rsi

  v1 = (_QWORD *)(*(_QWORD *)(a1 + 48) & 0xFFFFFFFFFFFFFFF8LL);
  if ( v1 == (_QWORD *)(a1 + 48) )
    goto LABEL_28;
  if ( !v1 )
    BUG();
  if ( (unsigned int)*((unsigned __int8 *)v1 - 24) - 30 > 0xA )
LABEL_28:
    BUG();
  if ( *((_BYTE *)v1 - 24) != 31 || (*((_DWORD *)v1 - 5) & 0x7FFFFFF) != 1 )
    return 0;
  v2 = *(_QWORD **)(a1 + 56);
  if ( v2 != v1 )
  {
    for ( i = (_QWORD *)(*v1 & 0xFFFFFFFFFFFFFFF8LL); ; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( !i )
        BUG();
      v4 = *((_BYTE *)i - 24);
      if ( v4 != 85 )
      {
        if ( v4 != 84 )
          return 0;
        goto LABEL_11;
      }
      v7 = *(i - 7);
      if ( !v7 )
        return 0;
      if ( *(_BYTE *)v7
        || *(_QWORD *)(v7 + 24) != i[7]
        || (*(_BYTE *)(v7 + 33) & 0x20) == 0
        || (unsigned int)(*(_DWORD *)(v7 + 36) - 68) > 3
        || v2 == i )
      {
        break;
      }
    }
    if ( *(_BYTE *)v7
      || *(_QWORD *)(v7 + 24) != i[7]
      || (*(_BYTE *)(v7 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v7 + 36) - 68) > 3 )
    {
      return 0;
    }
  }
LABEL_11:
  v5 = *(v1 - 7);
  if ( v5 )
  {
    if ( a1 == v5 )
      return 0;
  }
  if ( !(unsigned __int8)sub_2D64560(a1, *(v1 - 7)) )
    return 0;
  return v5;
}
