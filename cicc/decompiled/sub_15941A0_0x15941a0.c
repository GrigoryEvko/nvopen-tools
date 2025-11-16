// Function: sub_15941A0
// Address: 0x15941a0
//
__int64 __fastcall sub_15941A0(__int64 *a1)
{
  __int64 v1; // rdx

  v1 = *a1;
  if ( ((*(_BYTE *)(*a1 + 8) - 14) & 0xFD) != 0 )
    return *(unsigned int *)(v1 + 12);
  else
    return *(unsigned int *)(v1 + 32);
}
