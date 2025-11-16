// Function: sub_39479B0
// Address: 0x39479b0
//
void __fastcall sub_39479B0(unsigned __int64 *a1)
{
  unsigned __int64 v1; // rdi
  __int64 v2; // rax
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi
  __int64 v5; // rax
  __int64 v6; // rbx
  unsigned __int64 v7; // r14
  unsigned __int64 *v8; // r15
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rdi
  unsigned __int64 v11; // rdi
  unsigned __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // r12
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // r13
  unsigned __int64 v20; // r15
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r8
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  unsigned __int64 v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h]
  unsigned __int64 v32; // [rsp+28h] [rbp-48h]
  unsigned __int64 *v33; // [rsp+30h] [rbp-40h]
  __int64 v34; // [rsp+30h] [rbp-40h]
  __int64 v35; // [rsp+38h] [rbp-38h]

  v29 = a1[1];
  v32 = *a1;
  if ( v29 != *a1 )
  {
    do
    {
      v1 = *(_QWORD *)(v32 + 8);
      if ( *(_DWORD *)(v32 + 20) )
      {
        v2 = *(unsigned int *)(v32 + 16);
        if ( (_DWORD)v2 )
        {
          v35 = 0;
          v30 = 8 * v2;
          do
          {
            v3 = *(_QWORD *)(v1 + v35);
            if ( v3 != -8 && v3 )
            {
              v4 = *(_QWORD *)(v3 + 8);
              if ( *(_DWORD *)(v3 + 20) )
              {
                v5 = *(unsigned int *)(v3 + 16);
                if ( (_DWORD)v5 )
                {
                  v6 = 0;
                  v31 = 8 * v5;
                  do
                  {
                    v7 = *(_QWORD *)(v4 + v6);
                    if ( v7 != -8 && v7 )
                    {
                      v8 = *(unsigned __int64 **)(v7 + 128);
                      v33 = *(unsigned __int64 **)(v7 + 136);
                      if ( v33 != v8 )
                      {
                        do
                        {
                          v9 = *v8;
                          if ( *v8 )
                          {
                            sub_16C93F0((_QWORD *)*v8);
                            j_j___libc_free_0(v9);
                          }
                          v8 += 2;
                        }
                        while ( v33 != v8 );
                        v8 = *(unsigned __int64 **)(v7 + 128);
                      }
                      if ( v8 )
                        j_j___libc_free_0((unsigned __int64)v8);
                      sub_3947930(v7 + 72);
                      v10 = *(_QWORD *)(v7 + 72);
                      if ( v10 != v7 + 120 )
                        j_j___libc_free_0(v10);
                      v11 = *(_QWORD *)(v7 + 48);
                      if ( v11 )
                        j_j___libc_free_0(v11);
                      v12 = *(_QWORD *)(v7 + 8);
                      if ( *(_DWORD *)(v7 + 20) )
                      {
                        v13 = *(unsigned int *)(v7 + 16);
                        if ( (_DWORD)v13 )
                        {
                          v14 = 8 * v13;
                          v15 = 0;
                          do
                          {
                            v16 = *(_QWORD *)(v12 + v15);
                            if ( v16 != -8 && v16 )
                            {
                              v34 = v14;
                              _libc_free(v16);
                              v12 = *(_QWORD *)(v7 + 8);
                              v14 = v34;
                            }
                            v15 += 8;
                          }
                          while ( v14 != v15 );
                        }
                      }
                      _libc_free(v12);
                      _libc_free(v7);
                      v4 = *(_QWORD *)(v3 + 8);
                    }
                    v6 += 8;
                  }
                  while ( v31 != v6 );
                }
              }
              _libc_free(v4);
              _libc_free(v3);
              v1 = *(_QWORD *)(v32 + 8);
            }
            v35 += 8;
          }
          while ( v30 != v35 );
        }
      }
      _libc_free(v1);
      v17 = *(_QWORD *)v32;
      if ( *(_QWORD *)v32 )
      {
        v18 = *(unsigned __int64 **)(v17 + 128);
        v19 = *(unsigned __int64 **)(v17 + 120);
        if ( v18 != v19 )
        {
          do
          {
            v20 = *v19;
            if ( *v19 )
            {
              sub_16C93F0((_QWORD *)*v19);
              j_j___libc_free_0(v20);
            }
            v19 += 2;
          }
          while ( v18 != v19 );
          v19 = *(unsigned __int64 **)(v17 + 120);
        }
        if ( v19 )
          j_j___libc_free_0((unsigned __int64)v19);
        sub_3947930(v17 + 64);
        v21 = *(_QWORD *)(v17 + 64);
        if ( v21 != v17 + 112 )
          j_j___libc_free_0(v21);
        v22 = *(_QWORD *)(v17 + 40);
        if ( v22 )
          j_j___libc_free_0(v22);
        v23 = *(_QWORD *)v17;
        if ( *(_DWORD *)(v17 + 12) )
        {
          v24 = *(unsigned int *)(v17 + 8);
          if ( (_DWORD)v24 )
          {
            v25 = 8 * v24;
            v26 = 0;
            do
            {
              v27 = *(_QWORD *)(v23 + v26);
              if ( v27 != -8 && v27 )
              {
                _libc_free(v27);
                v23 = *(_QWORD *)v17;
              }
              v26 += 8;
            }
            while ( v25 != v26 );
          }
        }
        _libc_free(v23);
        j_j___libc_free_0(v17);
      }
      v32 += 40LL;
    }
    while ( v29 != v32 );
    v32 = *a1;
  }
  if ( v32 )
    j_j___libc_free_0(v32);
}
