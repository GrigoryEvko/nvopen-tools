// Function: sub_861C20
// Address: 0x861c20
//
_QWORD *__fastcall sub_861C20(__int64 a1, FILE *a2)
{
  int i; // esi
  __int64 v4; // rcx
  _QWORD *result; // rax

  for ( i = dword_4F04C64; ; i = *(_DWORD *)(v4 + 552) )
  {
    v4 = qword_4F04C68[0] + 776LL * i;
    result = *(_QWORD **)(v4 + 608);
    if ( result )
      break;
LABEL_7:
    if ( !i )
      return result;
  }
  while ( result[1] != a1 )
  {
    result = (_QWORD *)*result;
    if ( !result )
      goto LABEL_7;
  }
  return (_QWORD *)sub_686C80(0x30Cu, a2, a1, result[2]);
}
