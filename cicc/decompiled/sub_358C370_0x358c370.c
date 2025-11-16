// Function: sub_358C370
// Address: 0x358c370
//
_QWORD *__fastcall sub_358C370(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  unsigned __int64 v4; // rdx
  _QWORD *v5; // rcx

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (_QWORD *)(a1 + 8);
  if ( !v2 )
    return (_QWORD *)(a1 + 8);
  v4 = *a2;
  v5 = (_QWORD *)(a1 + 8);
  do
  {
    while ( v2[4] >= v4 && (v2[4] != v4 || v2[5] >= a2[1]) )
    {
      v5 = v2;
      v2 = (_QWORD *)v2[2];
      if ( !v2 )
        goto LABEL_8;
    }
    v2 = (_QWORD *)v2[3];
  }
  while ( v2 );
LABEL_8:
  if ( v3 == v5 || v5[4] > v4 )
    return (_QWORD *)(a1 + 8);
  if ( v5[4] != v4 )
    return v5;
  if ( a2[1] >= v5[5] )
    return v5;
  return v3;
}
