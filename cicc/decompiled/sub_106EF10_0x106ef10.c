// Function: sub_106EF10
// Address: 0x106ef10
//
__int64 __fastcall sub_106EF10(__int64 a1, int a2)
{
  __int64 v2; // rax
  __int64 v3; // r8
  unsigned int v4; // edx
  int *v5; // rcx
  int v6; // edi
  int v8; // ecx
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 432);
  v3 = *(_QWORD *)(a1 + 416);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (37 * a2);
    v5 = (int *)(v3 + 24LL * v4);
    v6 = *v5;
    if ( *v5 == a2 )
      return *((_QWORD *)v5 + 1);
    v8 = 1;
    while ( v6 != -1 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (int *)(v3 + 24LL * v4);
      v6 = *v5;
      if ( *v5 == a2 )
        return *((_QWORD *)v5 + 1);
      v8 = v9;
    }
  }
  return *(_QWORD *)(v3 + 24 * v2 + 8);
}
