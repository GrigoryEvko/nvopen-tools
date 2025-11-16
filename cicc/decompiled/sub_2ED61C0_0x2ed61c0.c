// Function: sub_2ED61C0
// Address: 0x2ed61c0
//
void __fastcall sub_2ED61C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 v9; // rbx
  __int64 *v10; // r15
  unsigned __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  char **v17; // rsi
  __int64 v18; // rdi
  __int64 v19; // r13
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  __int64 *v22; // rbx
  unsigned __int64 v23; // rdi
  __int64 *v24; // rax
  __int64 *v25; // rbx
  unsigned __int64 v26; // rdi
  __int64 v27; // r13
  __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 *v30; // r15
  char **v31; // rbx
  __int64 *v32; // r14
  __int64 v33; // rdx
  char **v34; // rsi
  __int64 v35; // rdi
  __int64 *v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r8
  __int64 v41; // r9
  int v42; // r14d
  __int64 v43; // rdx
  __int64 *v44; // rbx
  char **v45; // r14
  __int64 v46; // rcx
  char **v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // [rsp-60h] [rbp-60h]
  __int64 v50; // [rsp-58h] [rbp-58h]
  unsigned __int64 v51; // [rsp-58h] [rbp-58h]
  __int64 *v52; // [rsp-58h] [rbp-58h]
  int v53; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 v54; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(__int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v53 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v9 )
      {
        v24 = *(__int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v43 = 32 * v11;
          v44 = v10 + 1;
          v45 = (char **)(a2 + 24);
          v49 = v43;
          v52 = &v10[(unsigned __int64)v43 / 8 + 1];
          do
          {
            v46 = (__int64)*(v45 - 1);
            v47 = v45;
            v48 = (__int64)v44;
            v45 += 4;
            v44 += 4;
            *(v44 - 5) = v46;
            sub_2ED1840(v48, v47, v43, v46, a5, a6);
          }
          while ( v52 != v44 );
          v24 = *(__int64 **)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = (__int64 *)((char *)v10 + v49);
        }
        v25 = &v24[4 * v9];
        while ( v10 != v25 )
        {
          v25 -= 4;
          v26 = v25[1];
          if ( (__int64 *)v26 != v25 + 3 )
            _libc_free(v26);
        }
        *(_DWORD *)(a1 + 8) = v53;
        v27 = *(_QWORD *)a2;
        v28 = *(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v28 )
        {
          do
          {
            v28 -= 32;
            v29 = *(_QWORD *)(v28 + 8);
            if ( v29 != v28 + 24 )
              _libc_free(v29);
          }
          while ( v27 != v28 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v36 = &v10[4 * v9];
          while ( v36 != v10 )
          {
            while ( 1 )
            {
              v36 -= 4;
              v37 = v36[1];
              if ( (__int64 *)v37 == v36 + 3 )
                break;
              v51 = v11;
              _libc_free(v37);
              v11 = v51;
              if ( v36 == v10 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          v10 = (__int64 *)sub_C8D7D0(a1, a1 + 16, v11, 0x20u, &v54, a6);
          sub_2ED60F0((__int64 **)a1, (__int64)v10, v38, v39, v40, v41);
          v42 = v54;
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v10;
          *(_DWORD *)(a1 + 12) = v42;
          v6 = *(_QWORD *)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v12 = *(_QWORD *)a2;
        }
        else
        {
          v12 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v30 = v10 + 1;
            v31 = (char **)(a2 + 24);
            v50 = 32LL * *(unsigned int *)(a1 + 8);
            v32 = &v30[(unsigned __int64)v50 / 8];
            do
            {
              v33 = (__int64)*(v31 - 1);
              v34 = v31;
              v35 = (__int64)v30;
              v30 += 4;
              v31 += 4;
              *(v30 - 5) = v33;
              sub_2ED1840(v35, v34, v33, a4, a5, a6);
            }
            while ( v30 != v32 );
            v6 = *(_QWORD *)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = (__int64 *)(v50 + *(_QWORD *)a1);
            v12 = *(_QWORD *)a2 + v50;
          }
        }
        v13 = 32 * v11;
        v14 = v12;
        v15 = v13 + v6;
        if ( v15 != v12 )
        {
          do
          {
            while ( 1 )
            {
              if ( v10 )
              {
                v16 = *(_QWORD *)v14;
                *((_DWORD *)v10 + 4) = 0;
                *((_DWORD *)v10 + 5) = 2;
                *v10 = v16;
                v10[1] = (__int64)(v10 + 3);
                if ( *(_DWORD *)(v14 + 16) )
                  break;
              }
              v14 += 32;
              v10 += 4;
              if ( v15 == v14 )
                goto LABEL_12;
            }
            v17 = (char **)(v14 + 8);
            v18 = (__int64)(v10 + 1);
            v14 += 32;
            v10 += 4;
            sub_2ED1840(v18, v17, v13, a4, a5, a6);
          }
          while ( v15 != v14 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v53;
        v19 = *(_QWORD *)a2;
        v20 = *(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v20 )
        {
          do
          {
            v20 -= 32;
            v21 = *(_QWORD *)(v20 + 8);
            if ( v21 != v20 + 24 )
              _libc_free(v21);
          }
          while ( v19 != v20 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v22 = &v10[4 * v9];
      if ( v22 != v10 )
      {
        do
        {
          v22 -= 4;
          v23 = v22[1];
          if ( (__int64 *)v23 != v22 + 3 )
            _libc_free(v23);
        }
        while ( v22 != v10 );
        v10 = *(__int64 **)a1;
      }
      if ( v10 != (__int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v10);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
