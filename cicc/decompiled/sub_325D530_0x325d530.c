// Function: sub_325D530
// Address: 0x325d530
//
__int64 __fastcall sub_325D530(int **a1, unsigned int *a2)
{
  unsigned int v2; // r8d
  int v3; // eax
  __int64 v4; // r9
  __int64 v5; // rcx
  int v6; // edi

  v2 = *a2;
  v3 = **a1;
  if ( !v3 )
    return 1;
  v4 = *((_QWORD *)*a1 + 1);
  v5 = 0;
  while ( 1 )
  {
    v6 = *(_DWORD *)(v4 + 4 * v5);
    if ( v6 >= 0 && ((unsigned int)v5 % v2 || v6 != (unsigned int)v5 / v2) )
      break;
    if ( ++v5 == v3 )
      return 1;
  }
  return 0;
}
