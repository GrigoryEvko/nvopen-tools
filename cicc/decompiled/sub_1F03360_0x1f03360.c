// Function: sub_1F03360
// Address: 0x1f03360
//
void __fastcall sub_1F03360(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  int v4; // r15d
  int v5; // r14d
  __int64 v6; // rdx
  _BYTE v7[49]; // [rsp+Fh] [rbp-31h] BYREF

  v3 = *(_QWORD *)(a1 + 40);
  v4 = *(_DWORD *)(v3 + 4LL * *(unsigned int *)(a2 + 192));
  v5 = *(_DWORD *)(v3 + 4LL * *(unsigned int *)(a3 + 192));
  v7[0] = 0;
  if ( v4 < v5 )
  {
    v6 = *(_QWORD *)(a1 + 72);
    if ( v6 )
      memset(*(void **)(a1 + 64), 0, 8 * v6);
    sub_1F03050(a1, a2, v5, v7);
    sub_1F02250(a1, (_QWORD *)(a1 + 64), v4, v5);
  }
}
