// Function: sub_1D5D650
// Address: 0x1d5d650
//
__int64 __fastcall sub_1D5D650(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rdx
  _QWORD *v3; // rsi
  _QWORD *i; // rax
  char v5; // cl
  __int64 v6; // r13
  __int64 v8; // rcx

  v1 = sub_157EBA0(a1);
  if ( *(_BYTE *)(v1 + 16) != 26 )
    return 0;
  v2 = v1;
  if ( (*(_DWORD *)(v1 + 20) & 0xFFFFFFF) != 1 )
    return 0;
  v3 = *(_QWORD **)(a1 + 48);
  if ( v3 != (_QWORD *)(v1 + 24) )
  {
    for ( i = (_QWORD *)(*(_QWORD *)(v1 + 24) & 0xFFFFFFFFFFFFFFF8LL); ; i = (_QWORD *)(*i & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( !i )
        BUG();
      v5 = *((_BYTE *)i - 8);
      if ( v5 != 78 )
        break;
      v8 = *(i - 6);
      if ( *(_BYTE *)(v8 + 16) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 || (unsigned int)(*(_DWORD *)(v8 + 36) - 35) > 3 )
        return 0;
      if ( v3 == i )
        goto LABEL_8;
    }
    if ( v5 != 77 )
      return 0;
  }
LABEL_8:
  v6 = *(_QWORD *)(v2 - 24);
  if ( v6 && a1 == v6 || !(unsigned __int8)sub_1D5CFF0(a1, *(_QWORD *)(v2 - 24)) )
    return 0;
  return v6;
}
