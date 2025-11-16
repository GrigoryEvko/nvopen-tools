// Function: sub_FEEF30
// Address: 0xfeef30
//
__int64 __fastcall sub_FEEF30(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rdx
  __int64 result; // rax
  __int64 v5; // rdi
  unsigned int v6; // ecx
  __int64 *v7; // rdx
  __int64 v8; // r8
  int v9; // edx
  int v10; // r10d

  v3 = *(_QWORD *)(a1 + 32) + 32LL * a3;
  result = *(unsigned int *)(v3 + 24);
  v5 = *(_QWORD *)(v3 + 8);
  if ( (_DWORD)result )
  {
    v6 = (result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * result) )
        return *((unsigned int *)v7 + 2);
    }
    else
    {
      v9 = 1;
      while ( v8 != -4096 )
      {
        v10 = v9 + 1;
        v6 = (result - 1) & (v9 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v9 = v10;
      }
    }
    return 0;
  }
  return result;
}
