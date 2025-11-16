// Function: sub_25F5BA0
// Address: 0x25f5ba0
//
char __fastcall sub_25F5BA0(__int64 a1, unsigned int a2)
{
  char result; // al
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 *v10; // r9
  int v11; // edx
  unsigned int v12; // eax
  __int64 *v13; // rdi
  __int64 v14; // r8
  int v15; // edi
  int v16; // r10d

  result = 0;
  if ( *(_DWORD *)a1 != a2 )
  {
    v3 = *(_QWORD *)(a1 + 8);
    v4 = *(_QWORD *)(v3 - 8);
    if ( *(_QWORD *)(a1 + 16) == *(_QWORD *)(v4 + 32LL * a2) )
    {
      v5 = *(_QWORD *)(a1 + 24);
      v6 = 32LL * *(unsigned int *)(v3 + 72) + 8LL * a2;
      v7 = *(_QWORD *)(v5 + 8);
      v8 = *(_QWORD *)(v4 + v6);
      v9 = *(unsigned int *)(v5 + 24);
      v10 = (__int64 *)(v7 + 8 * v9);
      if ( (_DWORD)v9 )
      {
        v11 = v9 - 1;
        v12 = v11 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v13 = (__int64 *)(v7 + 8LL * v12);
        v14 = *v13;
        if ( v8 == *v13 )
          return v10 == v13;
        v15 = 1;
        while ( v14 != -4096 )
        {
          v16 = v15 + 1;
          v12 = v11 & (v15 + v12);
          v13 = (__int64 *)(v7 + 8LL * v12);
          v14 = *v13;
          if ( v8 == *v13 )
            return v10 == v13;
          v15 = v16;
        }
      }
      return 1;
    }
  }
  return result;
}
