// Function: sub_1E40FE0
// Address: 0x1e40fe0
//
__int64 __fastcall sub_1E40FE0(__int64 a1, int a2, __int64 a3)
{
  __int64 v3; // rax

  if ( a2 == 1 )
    return 0;
  v3 = 1;
  while ( *(_QWORD *)(a1 + 40LL * (unsigned int)(v3 + 1) + 24) != a3 )
  {
    v3 = (unsigned int)(v3 + 2);
    if ( (_DWORD)v3 == a2 )
      return 0;
  }
  return *(unsigned int *)(a1 + 40 * v3 + 8);
}
