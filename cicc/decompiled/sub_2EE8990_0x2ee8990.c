// Function: sub_2EE8990
// Address: 0x2ee8990
//
__int64 __fastcall sub_2EE8990(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rdi
  unsigned int v9; // r13d
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // r12
  __int64 v13; // rax
  int v14; // r15d
  __int64 v15; // r14
  __int64 result; // rax
  __int64 v17; // r8
  __int64 v18; // rsi
  int v19; // edx
  __int64 v20; // rax
  void *v21; // r8
  __int64 v22; // [rsp+8h] [rbp-38h]

  v7 = a1[1];
  v8 = a1[55];
  v9 = *(_DWORD *)(v8 + 88);
  v10 = v7 + 88LL * *(int *)(a2 + 24);
  v11 = *(_QWORD *)v10;
  v12 = v9 * *(_DWORD *)(a2 + 24);
  if ( *(_QWORD *)v10 )
  {
    v13 = *(unsigned int *)(v11 + 24);
    v14 = *(_DWORD *)(v11 + 24);
    v22 = v7 + 88 * v13;
    *(_DWORD *)(v10 + 24) = *(_DWORD *)(v22 + 24) + *(_DWORD *)sub_2EE8230(v8, *(_QWORD *)v10, v22, 5 * v13, v11, a6);
    *(_DWORD *)(v10 + 16) = *(_DWORD *)(v22 + 16);
    v15 = sub_2EE8970((__int64)a1, v14);
    result = sub_2EE8550(a1[55], v14);
    v17 = result;
    if ( v9 )
    {
      for ( result = 0; result != v9; ++result )
      {
        v18 = (unsigned int)(v12 + result);
        v19 = *(_DWORD *)(v15 + 4 * result) + *(_DWORD *)(v17 + 4 * result);
        *(_DWORD *)(a1[51] + 4 * v18) = v19;
      }
    }
  }
  else
  {
    *(_DWORD *)(v10 + 24) = 0;
    *(_DWORD *)(v10 + 16) = *(_DWORD *)(a2 + 24);
    v20 = a1[51];
    v21 = (void *)(v20 + 4 * v12);
    result = v20 + 4 * (v9 + v12);
    if ( v21 != (void *)result )
      return (__int64)memset(v21, 0, 4LL * v9);
  }
  return result;
}
