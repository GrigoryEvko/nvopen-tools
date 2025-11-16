// Function: sub_15B0BB0
// Address: 0x15b0bb0
//
__int64 __fastcall sub_15B0BB0(unsigned __int8 *a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // r8

  v1 = *a1;
  if ( *a1 > 0xEu )
  {
    if ( (unsigned __int8)(v1 - 32) <= 1u || v1 == 17 )
      return *(_QWORD *)&a1[8 * (1LL - *((unsigned int *)a1 + 2))];
  }
  else if ( v1 > 0xAu )
  {
    return *(_QWORD *)&a1[8 * (1LL - *((unsigned int *)a1 + 2))];
  }
  if ( (unsigned int)v1 - 18 <= 1 || v1 == 20 )
    return *(_QWORD *)&a1[8 * (1LL - *((unsigned int *)a1 + 2))];
  if ( v1 == 31 )
    return *(_QWORD *)&a1[-8 * *((unsigned int *)a1 + 2)];
  v2 = 0;
  if ( v1 == 21 )
    return *(_QWORD *)&a1[-8 * *((unsigned int *)a1 + 2)];
  return v2;
}
