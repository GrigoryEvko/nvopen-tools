// Function: sub_E0F8D0
// Address: 0xe0f8d0
//
unsigned __int64 __fastcall sub_E0F8D0(__int64 a1)
{
  _BYTE *v1; // rcx
  _BYTE *v2; // rdx
  unsigned __int64 result; // rax

  v1 = *(_BYTE **)(a1 + 8);
  v2 = *(_BYTE **)a1;
  if ( v1 == *(_BYTE **)a1 || (unsigned __int8)(*v2 - 48) > 9u )
    return 0;
  result = 0;
  do
  {
    *(_QWORD *)a1 = ++v2;
    result = (char)*(v2 - 1) - 48 + 10 * result;
  }
  while ( v1 != v2 && (unsigned __int8)(*v2 - 48) <= 9u );
  if ( v1 - v2 < result )
    return 0;
  *(_QWORD *)a1 = &v2[result];
  return result;
}
