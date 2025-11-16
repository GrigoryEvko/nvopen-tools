// Function: sub_8D6740
// Address: 0x8d6740
//
_QWORD *__fastcall sub_8D6740(__int64 a1)
{
  char v1; // dl
  __int64 i; // rax

  v1 = *(_BYTE *)(a1 + 140);
  for ( i = a1; v1 == 12; v1 = *(_BYTE *)(i + 140) )
    i = *(_QWORD *)(i + 160);
  if ( v1 == 2 )
    return sub_8D6540(a1);
  if ( v1 == 3 )
  {
    if ( *(_BYTE *)(i + 160) == 2 )
      return sub_72C610(4u);
    return (_QWORD *)a1;
  }
  if ( v1 != 4 || *(_BYTE *)(i + 160) != 2 )
    return (_QWORD *)a1;
  return sub_72C7D0(4u);
}
