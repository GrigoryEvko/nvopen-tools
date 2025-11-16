// Function: sub_3574310
// Address: 0x3574310
//
__int64 __fastcall sub_3574310(__int64 a1, int a2)
{
  __int64 v2; // rcx
  int v3; // eax
  int v4; // eax
  unsigned int v5; // edx
  int v6; // edi
  int v8; // r8d

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 248LL);
  v3 = *(_DWORD *)(*(_QWORD *)a1 + 264LL);
  if ( v3 )
  {
    v4 = v3 - 1;
    v5 = v4 & (37 * a2);
    v6 = *(_DWORD *)(v2 + 4LL * v5);
    if ( a2 == v6 )
      return 1;
    v8 = 1;
    while ( v6 != -1 )
    {
      v5 = v4 & (v8 + v5);
      v6 = *(_DWORD *)(v2 + 4LL * v5);
      if ( v6 == a2 )
        return 1;
      ++v8;
    }
  }
  return 0;
}
