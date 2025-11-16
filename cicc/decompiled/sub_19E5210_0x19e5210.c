// Function: sub_19E5210
// Address: 0x19e5210
//
__int64 __fastcall sub_19E5210(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v3; // eax
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 *v6; // rdx
  __int64 v7; // rdi
  int v8; // edx
  int v9; // r9d

  result = *(unsigned int *)(a1 + 24);
  if ( (_DWORD)result )
  {
    v3 = result - 1;
    v4 = *(_QWORD *)(a1 + 8);
    v5 = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
      return *((unsigned int *)v6 + 2);
    }
    else
    {
      v8 = 1;
      while ( v7 != -8 )
      {
        v9 = v8 + 1;
        v5 = v3 & (v8 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          return *((unsigned int *)v6 + 2);
        v8 = v9;
      }
      return 0;
    }
  }
  return result;
}
