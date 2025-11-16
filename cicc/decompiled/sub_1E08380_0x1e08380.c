// Function: sub_1E08380
// Address: 0x1e08380
//
__int64 __fastcall sub_1E08380(__int64 a1, __int64 a2)
{
  void *v3; // rdi
  unsigned int v4; // eax
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // r15
  __int64 v9; // r14
  _QWORD *v10; // r12
  _QWORD *v11; // r14
  __int64 v12; // r8
  __int64 v13; // rdi
  __int64 v15; // [rsp+8h] [rbp-38h]

  ++*(_QWORD *)(a1 + 1016);
  *(_DWORD *)(a1 + 240) = 0;
  v3 = *(void **)(a1 + 1032);
  if ( v3 == *(void **)(a1 + 1024) )
    goto LABEL_6;
  v4 = 4 * (*(_DWORD *)(a1 + 1044) - *(_DWORD *)(a1 + 1048));
  v5 = *(unsigned int *)(a1 + 1040);
  if ( v4 < 0x20 )
    v4 = 32;
  if ( (unsigned int)v5 <= v4 )
  {
    memset(v3, -1, 8 * v5);
LABEL_6:
    *(_QWORD *)(a1 + 1044) = 0;
    goto LABEL_7;
  }
  sub_16CC920(a1 + 1016);
LABEL_7:
  v6 = sub_22077B0(80);
  v7 = v6;
  if ( v6 )
  {
    *(_QWORD *)(v6 + 24) = 0;
    *(_QWORD *)v6 = v6 + 16;
    *(_QWORD *)(v6 + 8) = 0x100000000LL;
    *(_QWORD *)(v6 + 32) = 0;
    *(_QWORD *)(v6 + 40) = 0;
    *(_DWORD *)(v6 + 48) = 0;
    *(_QWORD *)(v6 + 64) = 0;
    *(_BYTE *)(v6 + 72) = 0;
    *(_DWORD *)(v6 + 76) = 0;
  }
  v8 = *(_QWORD *)(a1 + 1312);
  *(_QWORD *)(a1 + 1312) = v6;
  if ( v8 )
  {
    v9 = *(unsigned int *)(v8 + 48);
    if ( (_DWORD)v9 )
    {
      v10 = *(_QWORD **)(v8 + 32);
      v11 = &v10[2 * v9];
      do
      {
        if ( *v10 != -16 && *v10 != -8 )
        {
          v12 = v10[1];
          if ( v12 )
          {
            v13 = *(_QWORD *)(v12 + 24);
            if ( v13 )
            {
              v15 = v10[1];
              j_j___libc_free_0(v13, *(_QWORD *)(v12 + 40) - v13);
              v12 = v15;
            }
            j_j___libc_free_0(v12, 56);
          }
        }
        v10 += 2;
      }
      while ( v11 != v10 );
    }
    j___libc_free_0(*(_QWORD *)(v8 + 32));
    if ( *(_QWORD *)v8 != v8 + 16 )
      _libc_free(*(_QWORD *)v8);
    j_j___libc_free_0(v8, 80);
    v7 = *(_QWORD *)(a1 + 1312);
  }
  *(_QWORD *)(v7 + 64) = a2;
  sub_1E07D70(v7, 0);
  return 0;
}
