// Function: sub_254BB00
// Address: 0x254bb00
//
__int64 __fastcall sub_254BB00(__int64 a1)
{
  __int64 v1; // rdx

  if ( *(_BYTE *)(a1 + 28) )
    v1 = *(unsigned int *)(a1 + 20);
  else
    v1 = *(unsigned int *)(a1 + 16);
  return *(_QWORD *)(a1 + 8) + 8 * v1;
}
