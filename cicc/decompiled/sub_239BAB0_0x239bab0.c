// Function: sub_239BAB0
// Address: 0x239bab0
//
void __fastcall sub_239BAB0(__int64 a1, __int64 a2)
{
  unsigned __int64 *v4; // rbx
  unsigned __int64 *v5; // r15
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r9
  int v8; // r14d
  char *v9; // rdx
  _QWORD *v10; // rax
  _QWORD *v11; // rsi
  unsigned __int64 *v12; // r14
  __int64 v13; // rbx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rdi
  unsigned __int64 *v16; // r14
  unsigned __int64 v17; // r15
  unsigned __int64 v18; // rdi
  unsigned __int64 *v19; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 v21; // r8
  unsigned __int64 v22; // rdi
  unsigned __int64 *v23; // r14
  __int64 v24; // rbx
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // r8
  unsigned __int64 v29; // rdi
  unsigned __int64 *j; // r15
  unsigned __int64 v31; // rbx
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rax
  unsigned __int64 v34; // r8
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // [rsp-50h] [rbp-50h]
  unsigned __int64 v37; // [rsp-48h] [rbp-48h]
  unsigned __int64 v38; // [rsp-48h] [rbp-48h]
  unsigned __int64 v39; // [rsp-48h] [rbp-48h]
  unsigned __int64 v40; // [rsp-48h] [rbp-48h]
  unsigned __int64 v41; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v42; // [rsp-40h] [rbp-40h]
  unsigned __int64 v43; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v44; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v45; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v4 = (unsigned __int64 *)(a2 + 16);
    v5 = *(unsigned __int64 **)a1;
    v6 = *(unsigned int *)(a1 + 8);
    v42 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v7 = *(unsigned int *)(a2 + 8);
      v8 = *(_DWORD *)(a2 + 8);
      if ( v7 <= v6 )
      {
        v19 = *(unsigned __int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v45 = &v5[v7];
          do
          {
            v33 = *v4;
            *v4 = 0;
            v34 = *v5;
            *v5 = v33;
            if ( v34 )
            {
              v35 = *(_QWORD *)(v34 + 24);
              if ( v35 != v34 + 40 )
              {
                v41 = v34;
                _libc_free(v35);
                v34 = v41;
              }
              j_j___libc_free_0(v34);
            }
            ++v4;
            ++v5;
          }
          while ( v5 != v45 );
          v19 = *(unsigned __int64 **)a1;
          v6 = *(unsigned int *)(a1 + 8);
        }
        for ( i = &v19[v6]; v5 != i; --i )
        {
          v21 = *(i - 1);
          if ( v21 )
          {
            v22 = *(_QWORD *)(v21 + 24);
            if ( v22 != v21 + 40 )
            {
              v43 = v21;
              _libc_free(v22);
              v21 = v43;
            }
            j_j___libc_free_0(v21);
          }
        }
        *(_DWORD *)(a1 + 8) = v8;
        v23 = *(unsigned __int64 **)a2;
        v24 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v24 )
        {
          do
          {
            v25 = *(_QWORD *)(v24 - 8);
            v24 -= 8;
            if ( v25 )
            {
              v26 = *(_QWORD *)(v25 + 24);
              if ( v26 != v25 + 40 )
                _libc_free(v26);
              j_j___libc_free_0(v25);
            }
          }
          while ( v23 != (unsigned __int64 *)v24 );
        }
      }
      else
      {
        if ( v7 > *(unsigned int *)(a1 + 12) )
        {
          for ( j = &v42[v6]; j != v42; --j )
          {
            v31 = *(j - 1);
            if ( v31 )
            {
              v32 = *(_QWORD *)(v31 + 24);
              if ( v32 != v31 + 40 )
              {
                v39 = v7;
                _libc_free(v32);
                v7 = v39;
              }
              v40 = v7;
              j_j___libc_free_0(v31);
              v7 = v40;
            }
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_B1B4E0(a1, v7);
          v4 = *(unsigned __int64 **)a2;
          v6 = 0;
          v7 = *(unsigned int *)(a2 + 8);
          v42 = *(unsigned __int64 **)a1;
          v9 = *(char **)a2;
        }
        else
        {
          v9 = (char *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v6 *= 8LL;
            v44 = (unsigned __int64 *)((char *)v5 + v6);
            do
            {
              v27 = *v4;
              *v4 = 0;
              v28 = *v5;
              *v5 = v27;
              if ( v28 )
              {
                v29 = *(_QWORD *)(v28 + 24);
                if ( v29 != v28 + 40 )
                {
                  v36 = v6;
                  v37 = v28;
                  _libc_free(v29);
                  v6 = v36;
                  v28 = v37;
                }
                v38 = v6;
                j_j___libc_free_0(v28);
                v6 = v38;
              }
              ++v4;
              ++v5;
            }
            while ( v5 != v44 );
            v4 = *(unsigned __int64 **)a2;
            v7 = *(unsigned int *)(a2 + 8);
            v42 = *(unsigned __int64 **)a1;
            v9 = (char *)(*(_QWORD *)a2 + v6);
          }
        }
        v10 = (unsigned __int64 *)((char *)v42 + v6);
        v11 = (_QWORD *)((char *)v10 + (char *)&v4[v7] - v9);
        if ( &v4[v7] != (unsigned __int64 *)v9 )
        {
          do
          {
            if ( v10 )
            {
              *v10 = *(_QWORD *)v9;
              *(_QWORD *)v9 = 0;
            }
            ++v10;
            v9 += 8;
          }
          while ( v10 != v11 );
        }
        *(_DWORD *)(a1 + 8) = v8;
        v12 = *(unsigned __int64 **)a2;
        v13 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v13 )
        {
          do
          {
            v14 = *(_QWORD *)(v13 - 8);
            v13 -= 8;
            if ( v14 )
            {
              v15 = *(_QWORD *)(v14 + 24);
              if ( v15 != v14 + 40 )
                _libc_free(v15);
              j_j___libc_free_0(v14);
            }
          }
          while ( v12 != (unsigned __int64 *)v13 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v16 = &v5[v6];
      if ( v16 != v5 )
      {
        do
        {
          v17 = *--v16;
          if ( v17 )
          {
            v18 = *(_QWORD *)(v17 + 24);
            if ( v18 != v17 + 40 )
              _libc_free(v18);
            j_j___libc_free_0(v17);
          }
        }
        while ( v16 != v42 );
        v5 = *(unsigned __int64 **)a1;
      }
      if ( v5 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v5);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v4;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
