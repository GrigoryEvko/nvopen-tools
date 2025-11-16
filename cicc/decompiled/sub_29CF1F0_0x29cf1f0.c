// Function: sub_29CF1F0
// Address: 0x29cf1f0
//
__int64 __fastcall sub_29CF1F0(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  int v4; // eax
  int v5; // ecx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  __int64 v10; // rax
  int v11; // eax
  int v12; // r9d

  v2 = a1[6];
  if ( v2 == a1[7] )
  {
    v10 = *(_QWORD *)(a1[9] - 8LL) + 512LL;
    v3 = *(_QWORD *)(*(_QWORD *)(a1[9] - 8LL) + 488LL);
    v4 = *(_DWORD *)(v10 - 8);
    if ( !v4 )
      return 0;
  }
  else
  {
    v3 = *(_QWORD *)(v2 - 24);
    v4 = *(_DWORD *)(v2 - 8);
    if ( !v4 )
      return 0;
  }
  v5 = v4 - 1;
  v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v3 + 16LL * v6);
  v8 = *v7;
  if ( a2 == *v7 )
    return v7[1];
  v11 = 1;
  while ( v8 != -4096 )
  {
    v12 = v11 + 1;
    v6 = v5 & (v11 + v6);
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      return v7[1];
    v11 = v12;
  }
  return 0;
}
