// Function: sub_B538E0
// Address: 0xb538e0
//
__int64 __fastcall sub_B538E0(unsigned int *a1)
{
  __int64 result; // rax

  result = *a1;
  if ( *((_BYTE *)a1 + 4) )
    return sub_B52E90(result);
  return result;
}
