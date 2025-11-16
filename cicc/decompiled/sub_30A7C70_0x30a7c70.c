// Function: sub_30A7C70
// Address: 0x30a7c70
//
_QWORD *__fastcall sub_30A7C70(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rsi
  _QWORD *v4; // rax
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx
  unsigned __int64 v8; // r8

  v3 = sub_30A7A60(a2);
  v4 = *(_QWORD **)(a1 + 112);
  if ( !v4 )
    return 0;
  v5 = (_QWORD *)(a1 + 104);
  do
  {
    while ( 1 )
    {
      v6 = v4[2];
      v7 = v4[3];
      if ( v3 <= v4[4] )
        break;
      v4 = (_QWORD *)v4[3];
      if ( !v7 )
        goto LABEL_6;
    }
    v5 = v4;
    v4 = (_QWORD *)v4[2];
  }
  while ( v6 );
LABEL_6:
  v8 = 0;
  if ( (_QWORD *)(a1 + 104) != v5 )
  {
    v8 = v5[4];
    if ( v3 < v8 )
      return v4;
  }
  return (_QWORD *)v8;
}
