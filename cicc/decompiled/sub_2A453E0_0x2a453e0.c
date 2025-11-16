// Function: sub_2A453E0
// Address: 0x2a453e0
//
__int64 __fastcall sub_2A453E0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r8
  int v4; // ecx
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // r9
  int v9; // eax
  int v10; // r10d

  v2 = *(_DWORD *)(a1 + 1608);
  v3 = *(_QWORD *)(a1 + 1592);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v3 + 16LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      return *(_QWORD *)(a1 + 32) + 48LL * *((unsigned int *)v6 + 2);
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = v4 & (v9 + v5);
      v6 = (__int64 *)(v3 + 16LL * v5);
      v7 = *v6;
      if ( a2 == *v6 )
        return *(_QWORD *)(a1 + 32) + 48LL * *((unsigned int *)v6 + 2);
      v9 = v10;
    }
  }
  return *(_QWORD *)(a1 + 32);
}
