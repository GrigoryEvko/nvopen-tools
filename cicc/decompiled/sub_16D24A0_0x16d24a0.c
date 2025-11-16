// Function: sub_16D24A0
// Address: 0x16d24a0
//
__int64 __fastcall sub_16D24A0(__int64 *a1, char a2, unsigned __int64 a3)
{
  __int64 result; // rax
  unsigned __int64 v6; // rdx
  __int64 v7; // rcx

  result = -1;
  v6 = a1[1];
  if ( a3 < v6 )
  {
    v7 = *a1;
    result = a3;
    while ( *(_BYTE *)(v7 + result) == a2 )
    {
      if ( v6 == ++result )
        return -1;
    }
  }
  return result;
}
