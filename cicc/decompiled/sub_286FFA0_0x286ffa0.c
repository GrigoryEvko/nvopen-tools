// Function: sub_286FFA0
// Address: 0x286ffa0
//
_QWORD *__fastcall sub_286FFA0(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rbx
  unsigned __int64 v4; // rsi
  unsigned __int64 v5; // rcx
  _QWORD *v6; // rax
  __int64 v8; // rax

  v3 = *(_QWORD **)(a1 + 16);
  if ( v3 )
  {
    v4 = *(_QWORD *)(a2 + 16);
    while ( 1 )
    {
      v5 = v3[6];
      v6 = (_QWORD *)v3[3];
      if ( v5 > v4 )
        v6 = (_QWORD *)v3[2];
      if ( !v6 )
        break;
      v3 = v6;
    }
    if ( v4 >= v5 )
      goto LABEL_8;
  }
  else
  {
    v3 = (_QWORD *)(a1 + 8);
  }
  if ( *(_QWORD **)(a1 + 24) == v3 )
    return 0;
  v8 = sub_220EF80((__int64)v3);
  v4 = *(_QWORD *)(a2 + 16);
  v5 = *(_QWORD *)(v8 + 48);
  v3 = (_QWORD *)v8;
LABEL_8:
  if ( v4 > v5 )
    return 0;
  return v3;
}
