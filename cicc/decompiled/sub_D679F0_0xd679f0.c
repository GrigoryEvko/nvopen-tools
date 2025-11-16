// Function: sub_D679F0
// Address: 0xd679f0
//
bool __fastcall sub_D679F0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rdi
  _QWORD *v4; // rsi
  __int64 v6; // rcx
  __int64 v7; // rdx
  __int64 *v8; // r9
  int v9; // edx
  unsigned int v10; // eax
  __int64 *v11; // rdi
  __int64 v12; // r8
  int v13; // edi
  int v14; // r10d
  __int64 v15; // [rsp+8h] [rbp-8h] BYREF

  v2 = *a1;
  v15 = a2;
  if ( !*(_DWORD *)(v2 + 16) )
  {
    v3 = *(_QWORD **)(v2 + 32);
    v4 = &v3[*(unsigned int *)(v2 + 40)];
    return v4 != sub_D67930(v3, (__int64)v4, &v15);
  }
  v6 = *(_QWORD *)(v2 + 8);
  v7 = *(unsigned int *)(v2 + 24);
  v8 = (__int64 *)(v6 + 8 * v7);
  if ( (_DWORD)v7 )
  {
    v9 = v7 - 1;
    v10 = v9 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v11 = (__int64 *)(v6 + 8LL * v10);
    v12 = *v11;
    if ( a2 == *v11 )
      return v8 != v11;
    v13 = 1;
    while ( v12 != -4096 )
    {
      v14 = v13 + 1;
      v10 = v9 & (v13 + v10);
      v11 = (__int64 *)(v6 + 8LL * v10);
      v12 = *v11;
      if ( a2 == *v11 )
        return v8 != v11;
      v13 = v14;
    }
  }
  return 0;
}
