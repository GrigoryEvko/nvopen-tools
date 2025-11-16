// Function: sub_2367990
// Address: 0x2367990
//
__int64 __fastcall sub_2367990(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // r12
  __int64 v12; // rbx
  unsigned __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  __int64 v17; // r15
  __int64 v18; // rdx
  unsigned __int64 v19; // r14
  unsigned __int64 *v20; // rbx
  int v21; // ebx
  __int64 v23; // [rsp+0h] [rbp-60h]
  unsigned __int64 v24; // [rsp+18h] [rbp-48h]
  unsigned __int64 v25[7]; // [rsp+28h] [rbp-38h] BYREF

  v6 = sub_C8D7D0(a1, a1 + 16, a2, 0x1518u, v25, a6);
  v11 = *(_QWORD *)a1;
  v23 = v6;
  v12 = v6;
  v13 = *(_QWORD *)a1 + 5400LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    do
    {
      if ( v12 )
      {
        v14 = *(_QWORD *)v11;
        *(_DWORD *)(v12 + 16) = 0;
        *(_DWORD *)(v12 + 20) = 8;
        *(_QWORD *)v12 = v14;
        *(_QWORD *)(v12 + 8) = v12 + 24;
        if ( *(_DWORD *)(v11 + 16) )
          sub_2367460(v12 + 8, v11 + 8, v7, v8, v9, v10);
      }
      v11 += 5400LL;
      v12 += 5400;
    }
    while ( v13 != v11 );
    v24 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 5400LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v15 = *(unsigned int *)(v13 - 5384);
        v16 = *(_QWORD *)(v13 - 5392);
        v13 -= 5400LL;
        v15 *= 672;
        v17 = v16 + v15;
        if ( v16 != v16 + v15 )
        {
          do
          {
            v18 = *(unsigned int *)(v17 - 648);
            v19 = *(_QWORD *)(v17 - 656);
            v17 -= 672;
            v18 *= 160;
            v20 = (unsigned __int64 *)(v19 + v18);
            if ( v19 != v19 + v18 )
            {
              do
              {
                v20 -= 20;
                if ( (unsigned __int64 *)*v20 != v20 + 2 )
                  _libc_free(*v20);
              }
              while ( (unsigned __int64 *)v19 != v20 );
              v19 = *(_QWORD *)(v17 + 16);
            }
            if ( v19 != v17 + 32 )
              _libc_free(v19);
          }
          while ( v16 != v17 );
          v16 = *(_QWORD *)(v13 + 8);
        }
        if ( v16 != v13 + 24 )
          _libc_free(v16);
      }
      while ( v13 != v24 );
      v13 = *(_QWORD *)a1;
    }
  }
  v21 = v25[0];
  if ( a1 + 16 != v13 )
    _libc_free(v13);
  *(_DWORD *)(a1 + 12) = v21;
  *(_QWORD *)a1 = v23;
  return a1;
}
