// Function: sub_1E80550
// Address: 0x1e80550
//
__int64 __fastcall sub_1E80550(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rdi
  unsigned int v5; // r13d
  __int64 v6; // r14
  __int64 v7; // r8
  __int64 v8; // r12
  __int64 v9; // rax
  int v10; // r15d
  __int64 v11; // r14
  __int64 result; // rax
  __int64 v13; // r8
  __int64 v14; // rsi
  int v15; // edx
  __int64 v16; // rax
  void *v17; // r8
  __int64 v18; // [rsp+8h] [rbp-38h]

  v3 = a1[1];
  v4 = a1[55];
  v5 = *(_DWORD *)(v4 + 320);
  v6 = v3 + 88LL * *(int *)(a2 + 48);
  v7 = *(_QWORD *)v6;
  v8 = v5 * *(_DWORD *)(a2 + 48);
  if ( *(_QWORD *)v6 )
  {
    v9 = *(unsigned int *)(v7 + 48);
    v10 = *(_DWORD *)(v7 + 48);
    v18 = v3 + 88 * v9;
    *(_DWORD *)(v6 + 24) = *(_DWORD *)(v18 + 24) + *(_DWORD *)sub_1E7FE90(v4, *(_QWORD *)v6, v18, 5 * v9, v7);
    *(_DWORD *)(v6 + 16) = *(_DWORD *)(v18 + 16);
    v11 = sub_1E80530((__int64)a1, v10);
    result = sub_1E80160(a1[55], v10);
    v13 = result;
    if ( v5 )
    {
      for ( result = 0; result != v5; ++result )
      {
        v14 = (unsigned int)(v8 + result);
        v15 = *(_DWORD *)(v11 + 4 * result) + *(_DWORD *)(v13 + 4 * result);
        *(_DWORD *)(a1[51] + 4 * v14) = v15;
      }
    }
  }
  else
  {
    *(_DWORD *)(v6 + 24) = 0;
    *(_DWORD *)(v6 + 16) = *(_DWORD *)(a2 + 48);
    v16 = a1[51];
    v17 = (void *)(v16 + 4 * v8);
    result = v16 + 4 * (v5 + v8);
    if ( v17 != (void *)result )
      return (__int64)memset(v17, 0, 4LL * v5);
  }
  return result;
}
