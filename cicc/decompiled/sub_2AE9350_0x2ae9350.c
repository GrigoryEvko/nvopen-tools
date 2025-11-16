// Function: sub_2AE9350
// Address: 0x2ae9350
//
__int64 __fastcall sub_2AE9350(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v7; // r12
  _DWORD *v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  _DWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _DWORD *v16; // r15
  unsigned __int64 v17; // rdi
  int v18; // r15d
  __int64 v20; // [rsp+0h] [rbp-50h]
  __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v21 = a1 + 16;
  v20 = sub_C8D7D0(a1, a1 + 16, a2, 0x60u, v22, a6);
  v7 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v7 )
  {
    v8 = *(_DWORD **)a1;
    v9 = v20 + 48;
    v10 = v20;
    do
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = 0;
        v11 = (_DWORD *)(v10 + 16);
        *(_DWORD *)(v10 + 8) = 1;
        *(_DWORD *)(v10 + 12) = 0;
        do
        {
          if ( v11 )
            *v11 = -1;
          v11 += 2;
        }
        while ( (_DWORD *)v9 != v11 );
        sub_2AE9220((_DWORD *)v10, v8);
        *(_DWORD *)(v10 + 56) = 0;
        *(_QWORD *)(v10 + 48) = v10 + 64;
        *(_DWORD *)(v10 + 60) = 4;
        if ( v8[14] )
          sub_2AA8020(v9, (__int64)(v8 + 12), v12, v13, v14, v15);
      }
      v8 += 24;
      v10 += 96;
      v9 += 96;
    }
    while ( (_DWORD *)v7 != v8 );
    v16 = *(_DWORD **)a1;
    v7 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v7 )
    {
      do
      {
        v7 -= 96LL;
        v17 = *(_QWORD *)(v7 + 48);
        if ( v17 != v7 + 64 )
          _libc_free(v17);
        if ( (*(_BYTE *)(v7 + 8) & 1) == 0 )
          sub_C7D6A0(*(_QWORD *)(v7 + 16), 8LL * *(unsigned int *)(v7 + 24), 4);
      }
      while ( (_DWORD *)v7 != v16 );
      v7 = *(_QWORD *)a1;
    }
  }
  v18 = v22[0];
  if ( v21 != v7 )
    _libc_free(v7);
  *(_DWORD *)(a1 + 12) = v18;
  *(_QWORD *)a1 = v20;
  return v20;
}
