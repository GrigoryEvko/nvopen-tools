// Function: sub_728140
// Address: 0x728140
//
__int64 __fastcall sub_728140(_DWORD *a1, _DWORD *a2)
{
  unsigned int v2; // edx
  __int64 result; // rax

  v2 = a2[4];
  result = 1;
  if ( a1[4] >= v2 )
    return (unsigned int)-(a1[4] > v2);
  return result;
}
