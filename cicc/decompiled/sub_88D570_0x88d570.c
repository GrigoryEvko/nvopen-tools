// Function: sub_88D570
// Address: 0x88d570
//
__int64 __fastcall sub_88D570(__int64 *a1, __int64 *a2)
{
  int v2; // edx
  char v3; // al
  int v4; // ecx
  char v5; // al
  __int64 result; // rax

  if ( !a1 )
  {
    if ( !a2 )
      return 0;
    v2 = 0;
    goto LABEL_5;
  }
  v2 = 0;
  do
  {
    v3 = *((_BYTE *)a1 + 24);
    a1 = (__int64 *)*a1;
    v2 -= ((v3 & 8) == 0) - 1;
  }
  while ( a1 );
  if ( a2 )
  {
LABEL_5:
    v4 = 0;
    do
    {
      v5 = *((_BYTE *)a2 + 24);
      a2 = (__int64 *)*a2;
      v4 -= ((v5 & 8) == 0) - 1;
    }
    while ( a2 );
    goto LABEL_7;
  }
  v4 = 0;
LABEL_7:
  result = (unsigned int)-(v2 < v4);
  if ( v2 > v4 )
    return 1;
  return result;
}
