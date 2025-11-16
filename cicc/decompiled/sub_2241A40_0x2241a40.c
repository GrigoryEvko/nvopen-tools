// Function: sub_2241A40
// Address: 0x2241a40
//
unsigned __int64 __fastcall sub_2241A40(_QWORD *a1, char a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax
  unsigned __int64 v4; // rdx

  result = a3;
  v4 = a1[1];
  if ( result >= v4 )
    return -1;
  while ( *(_BYTE *)(*a1 + result) == a2 )
  {
    if ( ++result == v4 )
      return -1;
  }
  return result;
}
