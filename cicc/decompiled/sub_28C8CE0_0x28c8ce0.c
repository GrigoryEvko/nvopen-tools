// Function: sub_28C8CE0
// Address: 0x28c8ce0
//
__int64 __fastcall sub_28C8CE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r8
  int v4; // eax
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // rdi
  int v8; // edx
  int v9; // r9d

  result = *(unsigned int *)(a1 + 2440);
  v3 = *(_QWORD *)(a1 + 2424);
  if ( (_DWORD)result )
  {
    v4 = result - 1;
    v5 = v4 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
    {
      return *((unsigned int *)v6 + 2);
    }
    else
    {
      v8 = 1;
      while ( v7 != -4096 )
      {
        v9 = v8 + 1;
        v5 = v4 & (v8 + v5);
        v6 = (__int64 *)(v3 + 16LL * v5);
        v7 = *v6;
        if ( a2 == *v6 )
          return *((unsigned int *)v6 + 2);
        v8 = v9;
      }
      return 0;
    }
  }
  return result;
}
