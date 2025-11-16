// Function: sub_73E250
// Address: 0x73e250
//
_BYTE *__fastcall sub_73E250(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // rsi
  _BYTE *v8; // rax

  v6 = a1;
  if ( *(_BYTE *)(a1 + 24) )
  {
    if ( (*(_BYTE *)(a1 + 25) & 3) != 0 )
    {
      sub_7313A0(a1, a2, a3, a4, a5, a6);
      v7 = sub_72D600(*(_QWORD **)a1);
    }
    else
    {
      v6 = sub_731410(a1, 1);
      v7 = sub_72D600(*(_QWORD **)v6);
    }
    *(_QWORD *)(v6 + 16) = 0;
    v8 = sub_73DBF0(1u, v7, v6);
    v8[27] |= 2u;
    return v8;
  }
  return (_BYTE *)v6;
}
