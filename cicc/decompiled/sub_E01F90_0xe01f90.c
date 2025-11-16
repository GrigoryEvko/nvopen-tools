// Function: sub_E01F90
// Address: 0xe01f90
//
char __fastcall sub_E01F90(unsigned __int8 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 v5; // rdx
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  char result; // al

  if ( !(unsigned __int8)sub_E00080(a1) )
    return 3;
  if ( (*(_BYTE *)(a2 + 7) & 0x20) == 0 )
    return 3;
  v4 = sub_B91C10(a2, 1);
  if ( !v4 )
    return 3;
  if ( (*(_BYTE *)(a3 + 7) & 0x20) == 0 )
    return 3;
  v5 = sub_B91C10(a3, 1);
  if ( !v5 )
    return 3;
  result = sub_E01E90((__int64)a1, v4, v5, v6, v7, v8);
  if ( result )
    return 3;
  return result;
}
