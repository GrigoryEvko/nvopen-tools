// Function: sub_3947D10
// Address: 0x3947d10
//
void __fastcall sub_3947D10(__int64 a1)
{
  int v2; // esi
  unsigned __int64 v3; // rdi
  __int64 v4; // rax
  unsigned __int64 v5; // r13
  unsigned __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // rbx
  unsigned __int64 v9; // r15
  unsigned __int64 *v10; // r12
  unsigned __int64 *v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  unsigned __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r14
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // r12
  unsigned __int64 v22; // r13
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r8
  __int64 v26; // rax
  __int64 v27; // r12
  __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 v31; // [rsp+10h] [rbp-50h]
  __int64 v32; // [rsp+18h] [rbp-48h]
  __int64 v33; // [rsp+20h] [rbp-40h]
  unsigned __int64 v34; // [rsp+28h] [rbp-38h]
  __int64 v35; // [rsp+28h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 20);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *(unsigned int *)(a1 + 16);
    if ( (_DWORD)v4 )
    {
      v33 = 0;
      v31 = 8 * v4;
      do
      {
        v5 = *(_QWORD *)(v3 + v33);
        if ( v5 != -8 && v5 )
        {
          v6 = *(_QWORD *)(v5 + 8);
          if ( *(_DWORD *)(v5 + 20) )
          {
            v7 = *(unsigned int *)(v5 + 16);
            if ( (_DWORD)v7 )
            {
              v8 = 0;
              v32 = 8 * v7;
              do
              {
                v9 = *(_QWORD *)(v6 + v8);
                if ( v9 != -8 && v9 )
                {
                  v10 = *(unsigned __int64 **)(v9 + 136);
                  v11 = *(unsigned __int64 **)(v9 + 128);
                  if ( v10 != v11 )
                  {
                    do
                    {
                      if ( *v11 )
                      {
                        v34 = *v11;
                        sub_16C93F0((_QWORD *)*v11);
                        j_j___libc_free_0(v34);
                      }
                      v11 += 2;
                    }
                    while ( v10 != v11 );
                    v11 = *(unsigned __int64 **)(v9 + 128);
                  }
                  if ( v11 )
                    j_j___libc_free_0((unsigned __int64)v11);
                  sub_3947930(v9 + 72);
                  v12 = *(_QWORD *)(v9 + 72);
                  if ( v12 != v9 + 120 )
                    j_j___libc_free_0(v12);
                  v13 = *(_QWORD *)(v9 + 48);
                  if ( v13 )
                    j_j___libc_free_0(v13);
                  v14 = *(_QWORD *)(v9 + 8);
                  if ( *(_DWORD *)(v9 + 20) )
                  {
                    v15 = *(unsigned int *)(v9 + 16);
                    if ( (_DWORD)v15 )
                    {
                      v16 = 8 * v15;
                      v17 = 0;
                      do
                      {
                        v18 = *(_QWORD *)(v14 + v17);
                        if ( v18 && v18 != -8 )
                        {
                          v35 = v16;
                          _libc_free(v18);
                          v14 = *(_QWORD *)(v9 + 8);
                          v16 = v35;
                        }
                        v17 += 8;
                      }
                      while ( v16 != v17 );
                    }
                  }
                  _libc_free(v14);
                  _libc_free(v9);
                  v6 = *(_QWORD *)(v5 + 8);
                }
                v8 += 8;
              }
              while ( v32 != v8 );
            }
          }
          _libc_free(v6);
          _libc_free(v5);
          v3 = *(_QWORD *)(a1 + 8);
        }
        v33 += 8;
      }
      while ( v31 != v33 );
    }
  }
  _libc_free(v3);
  v19 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v20 = *(unsigned __int64 **)(v19 + 128);
    v21 = *(unsigned __int64 **)(v19 + 120);
    if ( v20 != v21 )
    {
      do
      {
        v22 = *v21;
        if ( *v21 )
        {
          sub_16C93F0((_QWORD *)*v21);
          j_j___libc_free_0(v22);
        }
        v21 += 2;
      }
      while ( v20 != v21 );
      v21 = *(unsigned __int64 **)(v19 + 120);
    }
    if ( v21 )
      j_j___libc_free_0((unsigned __int64)v21);
    sub_3947930(v19 + 64);
    v23 = *(_QWORD *)(v19 + 64);
    if ( v23 != v19 + 112 )
      j_j___libc_free_0(v23);
    v24 = *(_QWORD *)(v19 + 40);
    if ( v24 )
      j_j___libc_free_0(v24);
    v25 = *(_QWORD *)v19;
    if ( *(_DWORD *)(v19 + 12) )
    {
      v26 = *(unsigned int *)(v19 + 8);
      if ( (_DWORD)v26 )
      {
        v27 = 8 * v26;
        v28 = 0;
        do
        {
          v29 = *(_QWORD *)(v25 + v28);
          if ( v29 != -8 )
          {
            if ( v29 )
            {
              _libc_free(v29);
              v25 = *(_QWORD *)v19;
            }
          }
          v28 += 8;
        }
        while ( v27 != v28 );
      }
    }
    _libc_free(v25);
    j_j___libc_free_0(v19);
  }
}
