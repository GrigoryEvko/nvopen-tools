// Function: sub_30F5C00
// Address: 0x30f5c00
//
__int64 __fastcall sub_30F5C00(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  unsigned __int64 *v12; // r14
  __int64 v13; // r12
  unsigned __int64 *v14; // r13
  __int64 v15; // rsi
  __int64 v16; // rdi
  unsigned __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // r15
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  int v23; // r12d
  __int64 v25; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v26; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v27; // [rsp+18h] [rbp-48h]
  unsigned __int64 v28[7]; // [rsp+28h] [rbp-38h] BYREF

  v26 = (unsigned __int64 *)(a1 + 16);
  v7 = sub_C8D7D0(a1, a1 + 16, a2, 0x50u, v28, a6);
  v12 = *(unsigned __int64 **)a1;
  v25 = v7;
  v13 = v7;
  v14 = (unsigned __int64 *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
  if ( *(unsigned __int64 **)a1 != v14 )
  {
    do
    {
      while ( 1 )
      {
        if ( v13 )
        {
          *(_DWORD *)(v13 + 8) = 0;
          *(_QWORD *)v13 = v13 + 16;
          *(_DWORD *)(v13 + 12) = 8;
          if ( *((_DWORD *)v12 + 2) )
            break;
        }
        v12 += 10;
        v13 += 80;
        if ( v14 == v12 )
          goto LABEL_7;
      }
      v15 = (__int64)v12;
      v16 = v13;
      v12 += 10;
      v13 += 80;
      sub_30F57A0(v16, v15, v8, v9, v10, v11);
    }
    while ( v14 != v12 );
LABEL_7:
    v27 = *(unsigned __int64 **)a1;
    v14 = (unsigned __int64 *)(*(_QWORD *)a1 + 80LL * *(unsigned int *)(a1 + 8));
    if ( *(unsigned __int64 **)a1 != v14 )
    {
      do
      {
        v17 = *(v14 - 10);
        v18 = *((unsigned int *)v14 - 18);
        v14 -= 10;
        v19 = v17 + 8 * v18;
        if ( v17 != v19 )
        {
          do
          {
            v20 = *(_QWORD *)(v19 - 8);
            v19 -= 8LL;
            if ( v20 )
            {
              v21 = *(_QWORD *)(v20 + 64);
              if ( v21 != v20 + 80 )
                _libc_free(v21);
              v22 = *(_QWORD *)(v20 + 24);
              if ( v22 != v20 + 40 )
                _libc_free(v22);
              j_j___libc_free_0(v20);
            }
          }
          while ( v17 != v19 );
          v17 = *v14;
        }
        if ( (unsigned __int64 *)v17 != v14 + 2 )
          _libc_free(v17);
      }
      while ( v14 != v27 );
      v14 = *(unsigned __int64 **)a1;
    }
  }
  v23 = v28[0];
  if ( v26 != v14 )
    _libc_free((unsigned __int64)v14);
  *(_DWORD *)(a1 + 12) = v23;
  *(_QWORD *)a1 = v25;
  return v25;
}
