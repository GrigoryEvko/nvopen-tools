// Function: sub_30F57A0
// Address: 0x30f57a0
//
void __fastcall sub_30F57A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r14
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // r8
  char *v12; // rax
  _QWORD *v13; // rbx
  _QWORD *v14; // rcx
  unsigned __int64 *v15; // r14
  __int64 v16; // rbx
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // rax
  unsigned __int64 v21; // r15
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // rdi
  unsigned __int64 *v24; // rax
  unsigned __int64 *i; // rbx
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rdi
  unsigned __int64 v28; // rdi
  unsigned __int64 *v29; // r14
  __int64 v30; // rbx
  unsigned __int64 v31; // r13
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rax
  unsigned __int64 v35; // rbx
  unsigned __int64 v36; // rdi
  unsigned __int64 v37; // rdi
  unsigned __int64 *v38; // r15
  unsigned __int64 v39; // r14
  unsigned __int64 v40; // rdi
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rdx
  unsigned __int64 v43; // r8
  unsigned __int64 v44; // rdi
  unsigned __int64 v45; // rdi
  __int64 v46; // [rsp-58h] [rbp-58h]
  unsigned __int64 v47; // [rsp-50h] [rbp-50h]
  unsigned __int64 v48; // [rsp-50h] [rbp-50h]
  unsigned __int64 v49; // [rsp-50h] [rbp-50h]
  unsigned __int64 v50; // [rsp-50h] [rbp-50h]
  unsigned __int64 v51; // [rsp-50h] [rbp-50h]
  __int64 v52; // [rsp-48h] [rbp-48h]
  unsigned __int64 v53; // [rsp-48h] [rbp-48h]
  unsigned __int64 v54; // [rsp-48h] [rbp-48h]
  unsigned __int64 v55; // [rsp-48h] [rbp-48h]
  __int64 v56; // [rsp-48h] [rbp-48h]
  int v57; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v58; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v59; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v60; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = (unsigned __int64 *)(a2 + 16);
    v8 = *(unsigned __int64 **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v57 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v9 )
      {
        v24 = *(unsigned __int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v46 = v11;
          v56 = a2 + 8 * v11 + 16;
          do
          {
            v42 = *v6;
            *v6 = 0;
            v43 = *v10;
            *v10 = v42;
            if ( v43 )
            {
              v44 = *(_QWORD *)(v43 + 64);
              if ( v44 != v43 + 80 )
              {
                v50 = v43;
                _libc_free(v44);
                v43 = v50;
              }
              v45 = *(_QWORD *)(v43 + 24);
              if ( v45 != v43 + 40 )
              {
                v51 = v43;
                _libc_free(v45);
                v43 = v51;
              }
              j_j___libc_free_0(v43);
            }
            ++v6;
            ++v10;
          }
          while ( v6 != (unsigned __int64 *)v56 );
          v24 = *(unsigned __int64 **)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = &v8[v46];
        }
        for ( i = &v24[v9]; v10 != i; --i )
        {
          v26 = *(i - 1);
          if ( v26 )
          {
            v27 = *(_QWORD *)(v26 + 64);
            if ( v27 != v26 + 80 )
              _libc_free(v27);
            v28 = *(_QWORD *)(v26 + 24);
            if ( v28 != v26 + 40 )
              _libc_free(v28);
            j_j___libc_free_0(v26);
          }
        }
        *(_DWORD *)(a1 + 8) = v57;
        v29 = *(unsigned __int64 **)a2;
        v30 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v30 )
        {
          do
          {
            v31 = *(_QWORD *)(v30 - 8);
            v30 -= 8;
            if ( v31 )
            {
              v32 = *(_QWORD *)(v31 + 64);
              if ( v32 != v31 + 80 )
                _libc_free(v32);
              v33 = *(_QWORD *)(v31 + 24);
              if ( v33 != v31 + 40 )
                _libc_free(v33);
              j_j___libc_free_0(v31);
            }
          }
          while ( v29 != (unsigned __int64 *)v30 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v38 = &v8[v9];
          while ( v38 != v8 )
          {
            while ( 1 )
            {
              v39 = *--v38;
              if ( !v39 )
                break;
              v40 = *(_QWORD *)(v39 + 64);
              if ( v40 != v39 + 80 )
              {
                v53 = v11;
                _libc_free(v40);
                v11 = v53;
              }
              v41 = *(_QWORD *)(v39 + 24);
              if ( v41 != v39 + 40 )
              {
                v54 = v11;
                _libc_free(v41);
                v11 = v54;
              }
              v55 = v11;
              j_j___libc_free_0(v39);
              v11 = v55;
              if ( v38 == v8 )
                goto LABEL_67;
            }
          }
LABEL_67:
          *(_DWORD *)(a1 + 8) = 0;
          sub_30F56B0(a1, v11, v9, a4, v11, a6);
          v6 = *(unsigned __int64 **)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v9 = 0;
          v8 = *(unsigned __int64 **)a1;
          v12 = *(char **)a2;
        }
        else
        {
          v12 = (char *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v9 *= 8LL;
            v52 = a2 + v9 + 16;
            do
            {
              v34 = *v6;
              *v6 = 0;
              v35 = *v10;
              *v10 = v34;
              if ( v35 )
              {
                v36 = *(_QWORD *)(v35 + 64);
                if ( v36 != v35 + 80 )
                {
                  v47 = v9;
                  _libc_free(v36);
                  v9 = v47;
                }
                v37 = *(_QWORD *)(v35 + 24);
                if ( v37 != v35 + 40 )
                {
                  v48 = v9;
                  _libc_free(v37);
                  v9 = v48;
                }
                v49 = v9;
                j_j___libc_free_0(v35);
                v9 = v49;
              }
              ++v6;
              ++v10;
            }
            while ( v6 != (unsigned __int64 *)v52 );
            v6 = *(unsigned __int64 **)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v8 = *(unsigned __int64 **)a1;
            v12 = (char *)(*(_QWORD *)a2 + v9);
          }
        }
        v13 = (unsigned __int64 *)((char *)v8 + v9);
        v14 = (_QWORD *)((char *)v13 + (char *)&v6[v11] - v12);
        if ( &v6[v11] != (unsigned __int64 *)v12 )
        {
          do
          {
            if ( v13 )
            {
              *v13 = *(_QWORD *)v12;
              *(_QWORD *)v12 = 0;
            }
            ++v13;
            v12 += 8;
          }
          while ( v13 != v14 );
        }
        *(_DWORD *)(a1 + 8) = v57;
        v15 = *(unsigned __int64 **)a2;
        v16 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v16 )
        {
          do
          {
            v17 = *(_QWORD *)(v16 - 8);
            v16 -= 8;
            if ( v17 )
            {
              v18 = *(_QWORD *)(v17 + 64);
              if ( v18 != v17 + 80 )
                _libc_free(v18);
              v19 = *(_QWORD *)(v17 + 24);
              if ( v19 != v17 + 40 )
                _libc_free(v19);
              j_j___libc_free_0(v17);
            }
          }
          while ( v15 != (unsigned __int64 *)v16 );
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
            v22 = *(_QWORD *)(v21 + 64);
            if ( v22 != v21 + 80 )
            {
              v58 = v20;
              _libc_free(v22);
              v20 = v58;
            }
            v23 = *(_QWORD *)(v21 + 24);
            if ( v23 != v21 + 40 )
            {
              v59 = v20;
              _libc_free(v23);
              v20 = v59;
            }
            v60 = v20;
            j_j___libc_free_0(v21);
            v20 = v60;
          }
        }
        while ( v20 != v8 );
        v10 = *(unsigned __int64 **)a1;
      }
      if ( v10 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v10);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
