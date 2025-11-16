// Function: sub_392A7D0
// Address: 0x392a7d0
//
__int64 __fastcall sub_392A7D0(_QWORD *a1)
{
  unsigned __int8 *v1; // rdx

  v1 = (unsigned __int8 *)a1[18];
  if ( v1 == (unsigned __int8 *)(a1[19] + a1[20]) )
    return 0xFFFFFFFFLL;
  a1[18] = v1 + 1;
  return *v1;
}
