// Function: sub_2B425B0
// Address: 0x2b425b0
//
void __fastcall sub_2B425B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // r14
  unsigned __int64 *v8; // r15
  unsigned __int64 v9; // r12
  __int64 v10; // rcx
  unsigned __int64 v11; // rsi
  char **v12; // rax
  __int64 v13; // r12
  char **v14; // r15
  __int64 v15; // rdx
  char **v16; // rsi
  unsigned __int64 *v17; // r13
  unsigned __int64 *v18; // r12
  unsigned __int64 *v19; // r12
  __int64 v20; // rax
  unsigned __int64 *v21; // r12
  unsigned __int64 *v22; // r13
  unsigned __int64 *v23; // r12
  char *v24; // r15
  char **v25; // rsi
  unsigned __int64 *v26; // r12
  unsigned __int64 *v27; // r12
  char **v28; // rsi
  int v29; // [rsp-44h] [rbp-44h]
  __int64 v30; // [rsp-40h] [rbp-40h]
  __int64 v31; // [rsp-40h] [rbp-40h]
  __int64 v32; // [rsp-40h] [rbp-40h]
  __int64 v33; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = (char **)(a2 + 16);
    v8 = *(unsigned __int64 **)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v29 = v11;
      if ( v11 <= v9 )
      {
        v20 = *(_QWORD *)a1;
        if ( v11 )
        {
          v27 = &v8[4 * v11];
          do
          {
            v28 = v6;
            v33 = v10;
            v6 += 4;
            sub_2B0D510(v10, v28, a3, v10, a5, a6);
            v10 = v33 + 32;
          }
          while ( (unsigned __int64 *)(v33 + 32) != v27 );
          v20 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v21 = (unsigned __int64 *)(v20 + 32 * v9);
        while ( (unsigned __int64 *)v10 != v21 )
        {
          v21 -= 4;
          if ( (unsigned __int64 *)*v21 != v21 + 2 )
          {
            v31 = v10;
            _libc_free(*v21);
            v10 = v31;
          }
        }
        *(_DWORD *)(a1 + 8) = v29;
        v22 = *(unsigned __int64 **)a2;
        v23 = (unsigned __int64 *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v23 )
        {
          do
          {
            v23 -= 4;
            if ( (unsigned __int64 *)*v23 != v23 + 2 )
              _libc_free(*v23);
          }
          while ( v22 != v23 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v26 = &v8[4 * v9];
          while ( v8 != v26 )
          {
            while ( 1 )
            {
              v26 -= 4;
              if ( (unsigned __int64 *)*v26 == v26 + 2 )
                break;
              _libc_free(*v26);
              if ( v8 == v26 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          v9 = 0;
          sub_2B424B0(a1, v11, a3, v10, a5, a6);
          v6 = *(char ***)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v8 = *(unsigned __int64 **)a1;
          v12 = *(char ***)a2;
        }
        else
        {
          v12 = v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v9 *= 32LL;
            v24 = (char *)v8 + v9;
            do
            {
              v25 = v6;
              v32 = v10;
              v6 += 4;
              sub_2B0D510(v10, v25, a3, v10, a5, a6);
              v10 = v32 + 32;
            }
            while ( v24 != (char *)(v32 + 32) );
            v6 = *(char ***)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v8 = *(unsigned __int64 **)a1;
            v12 = (char **)(*(_QWORD *)a2 + v9);
          }
        }
        v13 = (__int64)v8 + v9;
        v14 = v12;
        v15 = (__int64)&v6[4 * v11];
        if ( (char **)v15 != v12 )
        {
          do
          {
            while ( 1 )
            {
              if ( v13 )
              {
                *(_DWORD *)(v13 + 8) = 0;
                *(_QWORD *)v13 = v13 + 16;
                *(_DWORD *)(v13 + 12) = 4;
                if ( *((_DWORD *)v14 + 2) )
                  break;
              }
              v14 += 4;
              v13 += 32;
              if ( (char **)v15 == v14 )
                goto LABEL_12;
            }
            v16 = v14;
            v30 = v15;
            v14 += 4;
            sub_2B0D510(v13, v16, v15, v10, a5, a6);
            v15 = v30;
            v13 += 32;
          }
          while ( (char **)v30 != v14 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v29;
        v17 = *(unsigned __int64 **)a2;
        v18 = (unsigned __int64 *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v18 )
        {
          do
          {
            v18 -= 4;
            if ( (unsigned __int64 *)*v18 != v18 + 2 )
              _libc_free(*v18);
          }
          while ( v17 != v18 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v19 = &v8[4 * v9];
      if ( v8 != v19 )
      {
        do
        {
          v19 -= 4;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
            _libc_free(*v19);
        }
        while ( v8 != v19 );
        v19 = *(unsigned __int64 **)a1;
      }
      if ( v19 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v19);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
