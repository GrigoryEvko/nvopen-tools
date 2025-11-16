// Function: sub_67D370
// Address: 0x67d370
//
_BOOL8 __fastcall sub_67D370(int *a1, unsigned __int8 a2, _DWORD *a3)
{
  _BOOL8 result; // rax
  char v5[12]; // [rsp+Ch] [rbp-14h] BYREF

  v5[0] = a2;
  if ( a2 > 7u )
    return 1;
  sub_67C4B0(a1, v5, a3);
  if ( v5[0] > 7u )
    return 1;
  result = 0;
  if ( v5[0] == 7 )
    return (unsigned int)sub_729F80((unsigned int)*a3) == 0;
  return result;
}
