// Function: sub_13D1570
// Address: 0x13d1570
//
__int64 __fastcall sub_13D1570(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  unsigned __int8 v4; // al
  _QWORD *v6; // r13

  v4 = *(_BYTE *)(a2 + 16);
  if ( *(_BYTE *)(a1 + 16) <= 0x10u )
  {
    if ( v4 <= 0x10u )
      return sub_1584C50(a1, a2, a3, a4, a3);
    goto LABEL_5;
  }
  v6 = (_QWORD *)a1;
  if ( v4 != 9 )
  {
LABEL_5:
    if ( v4 != 86 )
      return 0;
    v6 = *(_QWORD **)(a2 - 24);
    if ( *v6 != *(_QWORD *)a1 || *(_DWORD *)(a2 + 64) != a4 || 4 * a4 && memcmp(*(const void **)(a2 + 56), a3, 4 * a4) )
      return 0;
    if ( (_QWORD *)a1 != v6 )
      return 0;
  }
  return (__int64)v6;
}
