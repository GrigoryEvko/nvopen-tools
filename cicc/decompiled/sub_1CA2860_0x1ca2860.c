// Function: sub_1CA2860
// Address: 0x1ca2860
//
_BYTE *__fastcall sub_1CA2860(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6, unsigned __int8 a7)
{
  _BYTE *v7; // r10
  __int64 **v9; // rax
  char v11; // cl
  _QWORD *v12; // r8
  _QWORD *v13; // r11
  __int64 **v14; // rax

  v7 = (_BYTE *)a3;
  if ( *(_BYTE *)(a3 + 16) == 15 || sub_1C96F00(a3) )
  {
    v9 = (__int64 **)sub_1646BA0(*(__int64 **)(*(_QWORD *)v7 + 24LL), a6);
    return (_BYTE *)sub_1599A20(v9);
  }
  else if ( v11 == 9 )
  {
    v14 = (__int64 **)sub_1646BA0(*(__int64 **)(*(_QWORD *)v7 + 24LL), a6);
    return (_BYTE *)sub_1599EF0(v14);
  }
  else
  {
    return sub_1CA1B70(v13, a2, v7, a4, v12, a6, a7);
  }
}
