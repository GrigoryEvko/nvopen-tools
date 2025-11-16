// Function: sub_2C6EE00
// Address: 0x2c6ee00
//
__int64 __fastcall sub_2C6EE00(__int64 a1, __int64 *a2)
{
  unsigned int v2; // ecx
  __int64 v3; // rsi
  __int64 v4; // r8
  unsigned int v5; // eax
  __int64 *v6; // rdx
  __int64 v7; // rdi
  int v9; // edx
  int v10; // r10d

  v2 = *(_DWORD *)(a1 + 32);
  v3 = *a2;
  v4 = *(_QWORD *)(a1 + 16);
  if ( v2 )
  {
    v5 = (v2 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v6 = (__int64 *)(v4 + 88LL * v5);
    v7 = *v6;
    if ( v3 == *v6 )
      return (__int64)(v6 + 1);
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = (v2 - 1) & (v9 + v5);
      v6 = (__int64 *)(v4 + 88LL * v5);
      v7 = *v6;
      if ( v3 == *v6 )
        return (__int64)(v6 + 1);
      v9 = v10;
    }
  }
  return v4 + 88LL * v2 + 8;
}
