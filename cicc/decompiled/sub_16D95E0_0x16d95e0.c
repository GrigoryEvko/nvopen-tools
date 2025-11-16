// Function: sub_16D95E0
// Address: 0x16d95e0
//
void __fastcall sub_16D95E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rcx
  __int64 *v8; // rdi
  __int64 *v9; // rax
  __int64 *v10; // rbx
  __int64 *v11; // r13
  __int64 v12; // r14
  __int64 v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // r15
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rbx
  unsigned __int64 v22; // r13
  __int64 v23; // [rsp-48h] [rbp-48h]
  __int64 v24; // [rsp-40h] [rbp-40h]

  if ( a1 )
  {
    v7 = *(unsigned int *)(a1 + 8);
    v8 = *(__int64 **)a1;
    if ( (_DWORD)v7 )
    {
      if ( *v8 != -8 && *v8 )
      {
        v10 = v8;
      }
      else
      {
        v9 = v8 + 1;
        do
        {
          do
          {
            a3 = *v9;
            v10 = v9++;
          }
          while ( a3 == -8 );
        }
        while ( !a3 );
      }
      v7 = (unsigned int)v7;
      v11 = &v8[(unsigned int)v7];
      if ( v11 != v10 )
      {
        do
        {
          while ( 1 )
          {
            v12 = *(_QWORD *)(*v10 + 8);
            if ( v12 )
            {
              sub_16D9420(*(_QWORD **)(*v10 + 8), a2, a3, v7, a5, a6);
              a2 = 112;
              j_j___libc_free_0(v12, 112);
            }
            v13 = v10[1];
            a3 = (__int64)(v10 + 1);
            if ( !v13 || v13 == -8 )
              break;
            ++v10;
            if ( v11 == (__int64 *)a3 )
              goto LABEL_17;
          }
          v14 = v10 + 2;
          do
          {
            do
            {
              a3 = *v14;
              v10 = v14++;
            }
            while ( a3 == -8 );
          }
          while ( !a3 );
        }
        while ( v11 != v10 );
LABEL_17:
        v8 = *(__int64 **)a1;
      }
    }
    if ( *(_DWORD *)(a1 + 12) )
    {
      v15 = *(unsigned int *)(a1 + 8);
      if ( (_DWORD)v15 )
      {
        v16 = 0;
        v24 = 8 * v15;
        do
        {
          v17 = v8[v16 / 8];
          if ( v17 != -8 && v17 )
          {
            v18 = *(_QWORD *)(v17 + 16);
            if ( *(_DWORD *)(v17 + 28) )
            {
              v19 = *(unsigned int *)(v17 + 24);
              if ( (_DWORD)v19 )
              {
                v20 = 8 * v19;
                v21 = 0;
                do
                {
                  v22 = *(_QWORD *)(v18 + v21);
                  if ( v22 )
                  {
                    if ( v22 != -8 )
                    {
                      v23 = v20;
                      sub_16D93B0(v22 + 8, a2, v20, v7, a5, a6);
                      _libc_free(v22);
                      v18 = *(_QWORD *)(v17 + 16);
                      v20 = v23;
                    }
                  }
                  v21 += 8;
                }
                while ( v20 != v21 );
              }
            }
            _libc_free(v18);
            _libc_free(v17);
            v8 = *(__int64 **)a1;
          }
          v16 += 8LL;
        }
        while ( v24 != v16 );
      }
    }
    _libc_free((unsigned __int64)v8);
    j_j___libc_free_0(a1, 32);
  }
}
