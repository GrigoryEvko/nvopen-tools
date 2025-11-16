// Function: sub_2AAEE30
// Address: 0x2aaee30
//
_QWORD *__fastcall sub_2AAEE30(__int64 a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v8; // [rsp+18h] [rbp-18h]

  if ( !BYTE4(a2) && (_DWORD)a2 == 1 )
    return (_QWORD *)a1;
  v2 = *(_BYTE *)(a1 + 8);
  if ( v2 == 15 )
  {
    if ( !(unsigned __int8)sub_E45910(a1) )
      return (_QWORD *)a1;
  }
  else
  {
    if ( v2 == 7 )
    {
      v8 = a2;
      return (_QWORD *)sub_2AAEDF0(a1, v8);
    }
    if ( !(unsigned __int8)sub_BCBCB0(a1) )
      return (_QWORD *)a1;
  }
  v8 = a2;
  if ( *(_BYTE *)(a1 + 8) != 15 )
    return (_QWORD *)sub_2AAEDF0(a1, v8);
  return sub_E454C0(a1, a2, v3, v4, v5, v6);
}
