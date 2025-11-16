// Function: sub_EE35F0
// Address: 0xee35f0
//
__int64 __fastcall sub_EE35F0(_QWORD *a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  char *v6; // rcx

  *a2 = 0;
  if ( a1[1] == *a1 || (unsigned __int8)(*(_BYTE *)*a1 - 48) > 9u )
    return 1;
  v3 = 0;
  do
  {
    v4 = 10 * v3;
    v5 = -48;
    *a2 = v4;
    v6 = (char *)*a1;
    if ( *a1 != a1[1] )
    {
      *a1 = v6 + 1;
      v4 = *a2;
      v5 = *v6 - 48;
    }
    v3 = v4 + v5;
    *a2 = v3;
  }
  while ( a1[1] != *a1 && (unsigned __int8)(*(_BYTE *)*a1 - 48) <= 9u );
  return 0;
}
