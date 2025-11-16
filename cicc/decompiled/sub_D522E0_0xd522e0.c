// Function: sub_D522E0
// Address: 0xd522e0
//
_BYTE *__fastcall sub_D522E0(__int64 a1)
{
  _QWORD *v1; // rdx
  unsigned __int64 v2; // rax
  _BYTE *result; // rax

  v1 = (_QWORD *)(sub_D47930(a1) + 48);
  v2 = *v1 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v2 == v1 )
    goto LABEL_8;
  if ( !v2 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v2 - 24) - 30 > 0xA )
LABEL_8:
    BUG();
  if ( *(_BYTE *)(v2 - 24) != 31 )
    BUG();
  result = *(_BYTE **)(v2 - 120);
  if ( (unsigned __int8)(*result - 82) >= 2u )
    return 0;
  return result;
}
