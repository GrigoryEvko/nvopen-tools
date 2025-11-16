// Function: sub_2E6CC90
// Address: 0x2e6cc90
//
void __fastcall sub_2E6CC90(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rsi
  _QWORD *v8; // rax
  int v9; // r8d
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = *(_QWORD *)(a1 + 8);
  if ( v2 != a2 )
  {
    v3 = *(unsigned int *)(v2 + 32);
    v16[0] = a1;
    v6 = *(_QWORD **)(v2 + 24);
    v7 = (__int64)&v6[v3];
    v8 = sub_2E6C3E0(v6, v7, v16);
    v10 = (__int64)(v8 + 1);
    if ( v8 + 1 != (_QWORD *)v7 )
    {
      v11 = v7;
      v7 = (__int64)(v8 + 1);
      memmove(v8, v8 + 1, v11 - v10);
      v9 = *(_DWORD *)(v2 + 32);
    }
    v12 = (unsigned int)(v9 - 1);
    *(_DWORD *)(v2 + 32) = v12;
    *(_QWORD *)(a1 + 8) = a2;
    v13 = *(unsigned int *)(a2 + 32);
    v14 = *(unsigned int *)(a2 + 36);
    if ( v13 + 1 > v14 )
    {
      v7 = a2 + 40;
      sub_C8D5F0(a2 + 24, (const void *)(a2 + 40), v13 + 1, 8u, v12, v10);
      v13 = *(unsigned int *)(a2 + 32);
    }
    v15 = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(v15 + 8 * v13) = a1;
    ++*(_DWORD *)(a2 + 32);
    sub_2E6CB70(a1, v7, v15, v14, v12, v10);
  }
}
