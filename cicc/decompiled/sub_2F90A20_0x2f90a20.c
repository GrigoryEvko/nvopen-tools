// Function: sub_2F90A20
// Address: 0x2f90a20
//
void __fastcall sub_2F90A20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v4; // r15d
  int v5; // r14d
  void *v7; // rdi
  size_t v8; // rdx
  _BYTE v9[33]; // [rsp+Fh] [rbp-21h] BYREF

  v3 = *(_QWORD *)(a1 + 320);
  v4 = *(_DWORD *)(v3 + 4LL * *(unsigned int *)(a2 + 200));
  v5 = *(_DWORD *)(v3 + 4LL * *(unsigned int *)(a3 + 200));
  v9[0] = 0;
  if ( v4 < v5 )
  {
    v7 = *(void **)(a1 + 344);
    v8 = 8LL * *(unsigned int *)(a1 + 352);
    if ( v8 )
      memset(v7, 0, v8);
    sub_2F90810(a1, a2, v5, v9);
    sub_2F8FC90(a1, (_QWORD *)(a1 + 344), v4, v5);
  }
}
