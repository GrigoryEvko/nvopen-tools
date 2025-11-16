// Function: sub_23B6480
// Address: 0x23b6480
//
void __fastcall sub_23B6480(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  unsigned __int64 v4; // rdi
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // r14
  unsigned __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // r13
  __int64 v11; // r15
  _QWORD *v12; // r9
  unsigned __int64 v13; // rdi
  __int64 v14; // r10
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r12
  __int64 v19; // [rsp+0h] [rbp-50h]
  _QWORD *v20; // [rsp+8h] [rbp-48h]
  __int64 v21; // [rsp+10h] [rbp-40h]
  __int64 v22; // [rsp+18h] [rbp-38h]

  v2 = a1 + 64;
  v3 = *(_QWORD *)(a1 + 48);
  if ( v3 != v2 )
    j_j___libc_free_0(v3);
  v4 = *(_QWORD *)(a1 + 24);
  if ( *(_DWORD *)(a1 + 36) )
  {
    v5 = *(unsigned int *)(a1 + 32);
    if ( (_DWORD)v5 )
    {
      v6 = 0;
      v22 = 8 * v5;
      do
      {
        v7 = *(_QWORD *)(v4 + v6);
        if ( v7 != -8 && v7 )
        {
          v8 = *(_QWORD *)(v7 + 72);
          v21 = *(_QWORD *)v7 + 97LL;
          if ( *(_DWORD *)(v7 + 84) )
          {
            v9 = *(unsigned int *)(v7 + 80);
            if ( (_DWORD)v9 )
            {
              v10 = 8 * v9;
              v11 = 0;
              do
              {
                v12 = *(_QWORD **)(v8 + v11);
                if ( v12 != (_QWORD *)-8LL && v12 )
                {
                  v13 = v12[1];
                  v14 = *v12 + 41LL;
                  if ( (_QWORD *)v13 != v12 + 3 )
                  {
                    v19 = *v12 + 41LL;
                    v20 = v12;
                    j_j___libc_free_0(v13);
                    v14 = v19;
                    v12 = v20;
                  }
                  sub_C7D6A0((__int64)v12, v14, 8);
                  v8 = *(_QWORD *)(v7 + 72);
                }
                v11 += 8;
              }
              while ( v10 != v11 );
            }
          }
          _libc_free(v8);
          v15 = *(_QWORD *)(v7 + 40);
          if ( v15 != v7 + 56 )
            j_j___libc_free_0(v15);
          v16 = *(_QWORD *)(v7 + 8);
          if ( v16 != v7 + 24 )
            j_j___libc_free_0(v16);
          sub_C7D6A0(v7, v21, 8);
          v4 = *(_QWORD *)(a1 + 24);
        }
        v6 += 8;
      }
      while ( v22 != v6 );
    }
  }
  _libc_free(v4);
  v17 = *(unsigned __int64 **)(a1 + 8);
  v18 = *(unsigned __int64 **)a1;
  if ( v17 != *(unsigned __int64 **)a1 )
  {
    do
    {
      if ( (unsigned __int64 *)*v18 != v18 + 2 )
        j_j___libc_free_0(*v18);
      v18 += 4;
    }
    while ( v17 != v18 );
    v18 = *(unsigned __int64 **)a1;
  }
  if ( v18 )
    j_j___libc_free_0((unsigned __int64)v18);
}
