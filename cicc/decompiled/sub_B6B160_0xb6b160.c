// Function: sub_B6B160
// Address: 0xb6b160
//
__int64 __fastcall sub_B6B160(unsigned int a1, __int64 *a2)
{
  int *v2; // rax
  unsigned int v3; // r8d
  int v5; // edx
  __int64 v6; // rax

  v2 = (int *)a2[1];
  v3 = a1;
  if ( v2 )
  {
    if ( v2 == (int *)1 && (v5 = *(_DWORD *)*a2, v6 = *a2 + 12, a2[1] = 0, *a2 = v6, v5 == 1) )
      return a1 ^ 1;
    else
      return 1;
  }
  return v3;
}
