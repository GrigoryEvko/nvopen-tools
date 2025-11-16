// Function: sub_28AACA0
// Address: 0x28aaca0
//
__int64 __fastcall sub_28AACA0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // rdi
  int v9; // eax
  int v10; // r9d

  v2 = *(_DWORD *)(a1 + 56);
  v3 = *(_QWORD *)(a1 + 40);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      return v6[1];
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = v4 & (v9 + v5);
      v6 = (__int64 *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        return v6[1];
      v9 = v10;
    }
  }
  return 0;
}
