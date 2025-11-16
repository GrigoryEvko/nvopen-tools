// Function: sub_1E806A0
// Address: 0x1e806a0
//
unsigned __int64 __fastcall sub_1E806A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r13
  __int64 v9; // r12
  __int64 v10; // rdx
  _DWORD *v11; // r14
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // r8
  unsigned __int64 result; // rax
  __int64 v17; // rsi
  int v18; // edx
  size_t v19; // rdx
  unsigned int v20; // [rsp+Ch] [rbp-34h]

  v6 = a1[1];
  v7 = a1[55];
  v8 = v6 + 88LL * *(int *)(a2 + 48);
  v20 = *(_DWORD *)(v7 + 320);
  v9 = v20 * *(_DWORD *)(a2 + 48);
  *(_DWORD *)(v8 + 28) = *(_DWORD *)sub_1E7FE90(v7, a2, 11LL * *(int *)(a2 + 48), a4, a5);
  v11 = (_DWORD *)sub_1E80160(a1[55], *(_DWORD *)(a2 + 48));
  v12 = *(_QWORD *)(v8 + 8);
  if ( v12 )
  {
    v13 = *(unsigned int *)(v12 + 48);
    v14 = a1[1] + 88 * v13;
    *(_DWORD *)(v8 + 28) += *(_DWORD *)(v14 + 28);
    *(_DWORD *)(v8 + 20) = *(_DWORD *)(v14 + 20);
    v15 = sub_1E80680((__int64)a1, v13);
    result = v20;
    if ( v20 )
    {
      for ( result = 0; result != v20; ++result )
      {
        v17 = (unsigned int)(v9 + result);
        v18 = *(_DWORD *)(v15 + 4 * result) + v11[result];
        *(_DWORD *)(a1[53] + 4 * v17) = v18;
      }
    }
  }
  else
  {
    result = *(unsigned int *)(a2 + 48);
    v19 = 4 * v10;
    *(_DWORD *)(v8 + 20) = result;
    if ( v19 )
      return (unsigned __int64)memmove((void *)(a1[53] + 4 * v9), v11, v19);
  }
  return result;
}
