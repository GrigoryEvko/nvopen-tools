// Function: sub_3598140
// Address: 0x3598140
//
__int64 __fastcall sub_3598140(__int64 a1, __int64 a2)
{
  int v2; // ecx
  __int64 v3; // rdi
  __int64 v4; // rax

  v2 = *(_DWORD *)(a1 + 40) & 0xFFFFFF;
  if ( v2 == 1 )
    return 0;
  v3 = *(_QWORD *)(a1 + 32);
  v4 = 1;
  while ( *(_QWORD *)(v3 + 40LL * (unsigned int)(v4 + 1) + 24) == a2 )
  {
    v4 = (unsigned int)(v4 + 2);
    if ( v2 == (_DWORD)v4 )
      return 0;
  }
  return *(unsigned int *)(v3 + 40 * v4 + 8);
}
