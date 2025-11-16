// Function: sub_34BD060
// Address: 0x34bd060
//
void __fastcall sub_34BD060(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  unsigned __int64 *v7; // rbx
  unsigned __int64 *v8; // r15
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // r9
  int v11; // r14d
  unsigned __int64 v12; // rdx
  char *v13; // rdx
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  unsigned __int64 *v16; // r14
  __int64 v17; // rbx
  unsigned __int64 v18; // r13
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // r14
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // rdi
  unsigned __int64 *v27; // r14
  __int64 v28; // rbx
  unsigned __int64 v29; // r13
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // r8
  unsigned __int64 v33; // rdi
  __int64 v34; // rcx
  unsigned __int64 *j; // r15
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // r8
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // [rsp-50h] [rbp-50h]
  unsigned __int64 v42; // [rsp-48h] [rbp-48h]
  unsigned __int64 v43; // [rsp-48h] [rbp-48h]
  unsigned __int64 v44; // [rsp-48h] [rbp-48h]
  unsigned __int64 v45; // [rsp-48h] [rbp-48h]
  unsigned __int64 v46; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v47; // [rsp-40h] [rbp-40h]
  unsigned __int64 v48; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v49; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v50; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v7 = (unsigned __int64 *)(a2 + 16);
    v8 = *(unsigned __int64 **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v47 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v11 = *(_DWORD *)(a2 + 8);
      if ( v10 <= v9 )
      {
        v23 = *(unsigned __int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v50 = &v8[v10];
          do
          {
            v38 = *v7;
            *v7 = 0;
            v39 = *v8;
            *v8 = v38;
            if ( v39 )
            {
              v40 = *(_QWORD *)(v39 + 24);
              if ( v40 != v39 + 40 )
              {
                v46 = v39;
                _libc_free(v40);
                v39 = v46;
              }
              j_j___libc_free_0(v39);
            }
            ++v7;
            ++v8;
          }
          while ( v8 != v50 );
          v23 = *(unsigned __int64 **)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        for ( i = &v23[v9]; v8 != i; --i )
        {
          v25 = *(i - 1);
          if ( v25 )
          {
            v26 = *(_QWORD *)(v25 + 24);
            if ( v26 != v25 + 40 )
            {
              v48 = v25;
              _libc_free(v26);
              v25 = v48;
            }
            j_j___libc_free_0(v25);
          }
        }
        *(_DWORD *)(a1 + 8) = v11;
        v27 = *(unsigned __int64 **)a2;
        v28 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v28 )
        {
          do
          {
            v29 = *(_QWORD *)(v28 - 8);
            v28 -= 8;
            if ( v29 )
            {
              v30 = *(_QWORD *)(v29 + 24);
              if ( v30 != v29 + 40 )
                _libc_free(v30);
              j_j___libc_free_0(v29);
            }
          }
          while ( v27 != (unsigned __int64 *)v28 );
        }
      }
      else
      {
        v12 = *(unsigned int *)(a1 + 12);
        if ( v10 > v12 )
        {
          v34 = *(_QWORD *)a1;
          for ( j = &v47[v9]; j != v47; --j )
          {
            v36 = *(j - 1);
            if ( v36 )
            {
              v37 = *(_QWORD *)(v36 + 24);
              if ( v37 != v36 + 40 )
              {
                v44 = v10;
                _libc_free(v37);
                v10 = v44;
              }
              v45 = v10;
              j_j___libc_free_0(v36);
              v10 = v45;
            }
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_239B9C0(a1, v10, v12, v34, a5, v10);
          v7 = *(unsigned __int64 **)a2;
          v9 = 0;
          v10 = *(unsigned int *)(a2 + 8);
          v47 = *(unsigned __int64 **)a1;
          v13 = *(char **)a2;
        }
        else
        {
          v13 = (char *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v9 *= 8LL;
            v49 = (unsigned __int64 *)((char *)v8 + v9);
            do
            {
              v31 = *v7;
              *v7 = 0;
              v32 = *v8;
              *v8 = v31;
              if ( v32 )
              {
                v33 = *(_QWORD *)(v32 + 24);
                if ( v33 != v32 + 40 )
                {
                  v41 = v9;
                  v42 = v32;
                  _libc_free(v33);
                  v9 = v41;
                  v32 = v42;
                }
                v43 = v9;
                j_j___libc_free_0(v32);
                v9 = v43;
              }
              ++v7;
              ++v8;
            }
            while ( v8 != v49 );
            v7 = *(unsigned __int64 **)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v47 = *(unsigned __int64 **)a1;
            v13 = (char *)(*(_QWORD *)a2 + v9);
          }
        }
        v14 = (unsigned __int64 *)((char *)v47 + v9);
        v15 = (_QWORD *)((char *)v14 + (char *)&v7[v10] - v13);
        if ( &v7[v10] != (unsigned __int64 *)v13 )
        {
          do
          {
            if ( v14 )
            {
              *v14 = *(_QWORD *)v13;
              *(_QWORD *)v13 = 0;
            }
            ++v14;
            v13 += 8;
          }
          while ( v14 != v15 );
        }
        *(_DWORD *)(a1 + 8) = v11;
        v16 = *(unsigned __int64 **)a2;
        v17 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 - 8);
            v17 -= 8;
            if ( v18 )
            {
              v19 = *(_QWORD *)(v18 + 24);
              if ( v19 != v18 + 40 )
                _libc_free(v19);
              j_j___libc_free_0(v18);
            }
          }
          while ( v16 != (unsigned __int64 *)v17 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v20 = &v8[v9];
      if ( v20 != v8 )
      {
        do
        {
          v21 = *--v20;
          if ( v21 )
          {
            v22 = *(_QWORD *)(v21 + 24);
            if ( v22 != v21 + 40 )
              _libc_free(v22);
            j_j___libc_free_0(v21);
          }
        }
        while ( v20 != v47 );
        v8 = *(unsigned __int64 **)a1;
      }
      if ( v8 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v8);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v7;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
