// Function: sub_3950C20
// Address: 0x3950c20
//
__int64 __fastcall sub_3950C20(__int64 a1, __int64 *a2)
{
  __int64 v2; // rdx
  __int64 v3; // r8
  __int64 v4; // rsi
  unsigned int v5; // ecx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v9; // eax
  int v10; // r10d

  v2 = *(unsigned int *)(a1 + 32);
  v3 = *(_QWORD *)(a1 + 16);
  if ( (_DWORD)v2 )
  {
    v4 = *a2;
    v5 = (v2 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v6 = (__int64 *)(v3 + 40LL * v5);
    v7 = *v6;
    if ( v4 == *v6 )
      return (__int64)(v6 + 1);
    v9 = 1;
    while ( v7 != -8 )
    {
      v10 = v9 + 1;
      v5 = (v2 - 1) & (v9 + v5);
      v6 = (__int64 *)(v3 + 40LL * v5);
      v7 = *v6;
      if ( v4 == *v6 )
        return (__int64)(v6 + 1);
      v9 = v10;
    }
  }
  return v3 + 40 * v2 + 8;
}
