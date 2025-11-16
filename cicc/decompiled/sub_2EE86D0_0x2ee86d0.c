// Function: sub_2EE86D0
// Address: 0x2ee86d0
//
__int64 __fastcall sub_2EE86D0(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  int v4; // eax
  int v5; // ecx
  unsigned int v6; // edx
  __int64 *v7; // rax
  __int64 v8; // r8
  int v10; // eax
  int v11; // r9d

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 440) + 32LL);
  v3 = *(_QWORD *)(v2 + 8);
  v4 = *(_DWORD *)(v2 + 24);
  if ( v4 )
  {
    v5 = v4 - 1;
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v3 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      return v7[1];
    v10 = 1;
    while ( v8 != -4096 )
    {
      v11 = v10 + 1;
      v6 = v5 & (v10 + v6);
      v7 = (__int64 *)(v3 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        return v7[1];
      v10 = v11;
    }
  }
  return 0;
}
