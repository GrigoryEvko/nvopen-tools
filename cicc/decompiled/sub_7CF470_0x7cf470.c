// Function: sub_7CF470
// Address: 0x7cf470
//
_QWORD *__fastcall sub_7CF470(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v4; // rbx
  __int64 v5; // rdi

  if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
  {
    do
    {
      v4 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL);
      sub_7CED60(v4, a2, a3);
      a1 = v4;
    }
    while ( (*(_BYTE *)(v4 + 89) & 4) != 0 );
  }
  v5 = *(_QWORD *)(a1 + 40);
  if ( v5 )
  {
    if ( *(_BYTE *)(v5 + 28) == 3 )
      v5 = *(_QWORD *)(v5 + 32);
    else
      v5 = 0;
  }
  return sub_7CEBB0(v5, a2);
}
