// Function: sub_721630
// Address: 0x721630
//
_BYTE *__fastcall sub_721630(_BYTE *a1)
{
  _BYTE *v1; // rbx
  unsigned __int8 v2; // r12
  _BYTE *result; // rax
  bool v4; // zf

  v1 = a1;
  v2 = *a1;
  if ( !unk_4F07594 )
  {
LABEL_5:
    if ( v2 )
      goto LABEL_6;
    return v1;
  }
  if ( isalpha(v2) && a1[1] == 58 )
    goto LABEL_15;
  if ( v2 != 92 )
    goto LABEL_5;
  if ( a1[1] == 92 )
  {
LABEL_15:
    v2 = a1[2];
    v1 = a1 + 2;
    if ( !v2 )
      return v1;
  }
LABEL_6:
  result = v1;
  do
  {
    while ( 1 )
    {
      ++v1;
      if ( v2 != 92 || !unk_4F07598 )
        break;
      v2 = *v1;
      result = v1;
      if ( !*v1 )
        return result;
    }
    v4 = v2 == 47;
    v2 = *v1;
    if ( v4 )
      result = v1;
  }
  while ( v2 );
  return result;
}
