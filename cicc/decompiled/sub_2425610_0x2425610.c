// Function: sub_2425610
// Address: 0x2425610
//
__int64 __fastcall sub_2425610(__int64 a1, unsigned int *a2)
{
  _QWORD *v2; // r14
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v6; // rax

  *a2 = 0;
  v2 = *(_QWORD **)(a1 + 80);
  if ( v2 == (_QWORD *)(a1 + 72) )
    return 0;
  while ( 1 )
  {
    if ( !v2 )
      BUG();
    v3 = (_QWORD *)v2[4];
    if ( v3 != v2 + 3 )
      break;
LABEL_17:
    v2 = (_QWORD *)v2[1];
    if ( (_QWORD *)(a1 + 72) == v2 )
      return 0;
  }
  while ( 1 )
  {
    if ( !v3 )
      BUG();
    if ( (*((_BYTE *)v3 - 24) != 85
       || (v6 = *(v3 - 7)) == 0
       || *(_BYTE *)v6
       || *(_QWORD *)(v6 + 24) != v3[7]
       || (*(_BYTE *)(v6 + 33) & 0x20) == 0
       || (unsigned int)(*(_DWORD *)(v6 + 36) - 68) > 3)
      && v3[3]
      && (unsigned int)sub_B10CE0((__int64)(v3 + 3)) )
    {
      break;
    }
    v3 = (_QWORD *)v3[1];
    if ( v2 + 3 == v3 )
      goto LABEL_17;
  }
  v4 = sub_B10CE0((__int64)(v3 + 3));
  if ( *a2 >= v4 )
    v4 = *a2;
  *a2 = v4;
  return 1;
}
