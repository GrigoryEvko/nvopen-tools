// Function: sub_193EAF0
// Address: 0x193eaf0
//
__int64 __fastcall sub_193EAF0(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  unsigned int v4; // edi
  __int64 *v5; // rax
  __int64 v6; // rdx
  int v8; // eax
  int v9; // r10d

  v2 = *(unsigned int *)(a1 + 552);
  v3 = *(_QWORD *)(a1 + 536);
  if ( (_DWORD)v2 )
  {
    v4 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v5 = (__int64 *)(v3 + 16LL * v4);
    v6 = *v5;
    if ( a2 == *v5 )
      return *((unsigned int *)v5 + 2);
    v8 = 1;
    while ( v6 != -8 )
    {
      v9 = v8 + 1;
      v4 = (v2 - 1) & (v8 + v4);
      v5 = (__int64 *)(v3 + 16LL * v4);
      v6 = *v5;
      if ( a2 == *v5 )
        return *((unsigned int *)v5 + 2);
      v8 = v9;
    }
  }
  return *(unsigned int *)(v3 + 16 * v2 + 8);
}
