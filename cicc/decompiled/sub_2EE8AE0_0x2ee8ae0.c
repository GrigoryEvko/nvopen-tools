// Function: sub_2EE8AE0
// Address: 0x2ee8ae0
//
unsigned __int64 __fastcall sub_2EE8AE0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // rdx
  _DWORD *v12; // r14
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // r8
  unsigned __int64 result; // rax
  __int64 v18; // rsi
  int v19; // edx
  size_t v20; // rdx
  unsigned int v21; // [rsp+Ch] [rbp-34h]

  v7 = a1[1];
  v8 = a1[55];
  v9 = v7 + 88LL * *(int *)(a2 + 24);
  v21 = *(_DWORD *)(v8 + 88);
  v10 = v21 * *(_DWORD *)(a2 + 24);
  *(_DWORD *)(v9 + 28) = *(_DWORD *)sub_2EE8230(v8, a2, 11LL * *(int *)(a2 + 24), a4, a5, a6);
  v12 = (_DWORD *)sub_2EE8550(a1[55], *(_DWORD *)(a2 + 24));
  v13 = *(_QWORD *)(v9 + 8);
  if ( v13 )
  {
    v14 = *(unsigned int *)(v13 + 24);
    v15 = a1[1] + 88 * v14;
    *(_DWORD *)(v9 + 28) += *(_DWORD *)(v15 + 28);
    *(_DWORD *)(v9 + 20) = *(_DWORD *)(v15 + 20);
    v16 = sub_2EE8AC0((__int64)a1, v14);
    result = v21;
    if ( v21 )
    {
      for ( result = 0; result != v21; ++result )
      {
        v18 = (unsigned int)(v10 + result);
        v19 = *(_DWORD *)(v16 + 4 * result) + v12[result];
        *(_DWORD *)(a1[53] + 4 * v18) = v19;
      }
    }
  }
  else
  {
    result = *(unsigned int *)(a2 + 24);
    v20 = 4 * v11;
    *(_DWORD *)(v9 + 20) = result;
    if ( v20 )
      return (unsigned __int64)memmove((void *)(a1[53] + 4 * v10), v12, v20);
  }
  return result;
}
