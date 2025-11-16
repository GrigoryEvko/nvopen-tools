// Function: sub_A17370
// Address: 0xa17370
//
__int64 __fastcall sub_A17370(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rcx
  __int64 v4; // r8
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v8; // eax
  int v9; // r10d

  result = 0;
  if ( a2 )
  {
    v3 = *(unsigned int *)(a1 + 408);
    v4 = *(_QWORD *)(a1 + 392);
    if ( (_DWORD)v3 )
    {
      v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v6 = (__int64 *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        return *((unsigned int *)v6 + 2);
      v8 = 1;
      while ( v7 != -4 )
      {
        v9 = v8 + 1;
        v5 = (v3 - 1) & (v8 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          return *((unsigned int *)v6 + 2);
        v8 = v9;
      }
    }
    v6 = (__int64 *)(v4 + 16 * v3);
    return *((unsigned int *)v6 + 2);
  }
  return result;
}
