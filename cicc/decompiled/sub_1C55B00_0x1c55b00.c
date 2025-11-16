// Function: sub_1C55B00
// Address: 0x1c55b00
//
_QWORD *__fastcall sub_1C55B00(__int64 a1, unsigned __int64 *a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // r8
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rdi
  __int64 v6; // rcx
  __int64 v7; // rdx

  v2 = *(_QWORD **)(a1 + 16);
  v3 = (_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *a2;
    v5 = (_QWORD *)(a1 + 8);
    do
    {
      while ( 1 )
      {
        v6 = v2[2];
        v7 = v2[3];
        if ( v2[4] >= v4 )
          break;
        v2 = (_QWORD *)v2[3];
        if ( !v7 )
          goto LABEL_6;
      }
      v5 = v2;
      v2 = (_QWORD *)v2[2];
    }
    while ( v6 );
LABEL_6:
    if ( v5 != v3 && v5[4] <= v4 )
      return v5;
  }
  return v3;
}
