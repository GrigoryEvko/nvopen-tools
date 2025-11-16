// Function: sub_2F90B20
// Address: 0x2f90b20
//
__int64 __fastcall sub_2F90B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  int v9; // edx
  int v10; // r14d
  __int64 result; // rax
  __int64 v12; // rdx
  void *v13; // rdi
  size_t v14; // rdx
  _BYTE v15[33]; // [rsp+Fh] [rbp-21h] BYREF

  sub_2F90AB0(a1, a2, a3, a4, a5, a6);
  v8 = *(_QWORD *)(a1 + 320);
  v9 = *(_DWORD *)(v8 + 4LL * *(unsigned int *)(a3 + 200));
  v10 = *(_DWORD *)(v8 + 4LL * *(unsigned int *)(a2 + 200));
  v15[0] = 0;
  result = 0;
  if ( v9 < v10 )
  {
    v12 = *(unsigned int *)(a1 + 352);
    v13 = *(void **)(a1 + 344);
    v14 = 8 * v12;
    if ( v14 )
      memset(v13, 0, v14);
    sub_2F90810(a1, a3, v10, v15);
    return v15[0];
  }
  return result;
}
