// Function: sub_23672E0
// Address: 0x23672e0
//
__int64 __fastcall sub_23672E0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  __int64 v10; // r13
  unsigned __int64 v11; // rax
  __int64 v12; // rcx
  unsigned __int64 v13; // r12
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  unsigned __int64 v18; // r14
  unsigned __int64 *v19; // r15
  int v20; // r13d
  __int64 v22; // [rsp+8h] [rbp-58h]
  unsigned __int64 v23; // [rsp+18h] [rbp-48h]
  unsigned __int64 v24[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x2A0u, v24, a6);
  v9 = *(unsigned int *)(a1 + 8);
  v22 = v6;
  v10 = v6;
  v11 = *(_QWORD *)a1;
  v12 = 5 * v9;
  v13 = *(_QWORD *)a1 + 672 * v9;
  if ( *(_QWORD *)a1 != v13 )
  {
    do
    {
      if ( v10 )
      {
        *(_QWORD *)v10 = *(_QWORD *)v11;
        v14 = *(_QWORD *)(v11 + 8);
        *(_DWORD *)(v10 + 24) = 0;
        *(_QWORD *)(v10 + 8) = v14;
        *(_QWORD *)(v10 + 16) = v10 + 32;
        *(_DWORD *)(v10 + 28) = 4;
        v15 = *(unsigned int *)(v11 + 24);
        if ( (_DWORD)v15 )
        {
          v23 = v11;
          sub_2366F40(v10 + 16, v11 + 16, v15, v12, v7, v8);
          v11 = v23;
        }
      }
      v11 += 672LL;
      v10 += 672;
    }
    while ( v13 != v11 );
    v16 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 672LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v17 = *(unsigned int *)(v13 - 648);
        v18 = *(_QWORD *)(v13 - 656);
        v13 -= 672LL;
        v17 *= 160;
        v19 = (unsigned __int64 *)(v18 + v17);
        if ( v18 != v18 + v17 )
        {
          do
          {
            v19 -= 20;
            if ( (unsigned __int64 *)*v19 != v19 + 2 )
              _libc_free(*v19);
          }
          while ( (unsigned __int64 *)v18 != v19 );
          v18 = *(_QWORD *)(v13 + 16);
        }
        if ( v18 != v13 + 32 )
          _libc_free(v18);
      }
      while ( v13 != v16 );
      v13 = *(_QWORD *)a1;
    }
  }
  v20 = v24[0];
  if ( a1 + 16 != v13 )
    _libc_free(v13);
  *(_DWORD *)(a1 + 12) = v20;
  *(_QWORD *)a1 = v22;
  return v22;
}
