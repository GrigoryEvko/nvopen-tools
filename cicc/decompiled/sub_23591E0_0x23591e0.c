// Function: sub_23591E0
// Address: 0x23591e0
//
void __fastcall sub_23591E0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rsi
  int v12; // r15d
  __int64 v13; // rdx
  __int64 v14; // rcx
  unsigned __int64 v15; // rbx
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  char **v20; // rsi
  __int64 v21; // r13
  __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdx
  unsigned __int64 v27; // rbx
  unsigned __int64 v28; // rdi
  __int64 v29; // r13
  __int64 v30; // rbx
  unsigned __int64 v31; // rdi
  __int64 v32; // r14
  char **v33; // rdx
  __int64 v34; // rdi
  __int64 v35; // rdx
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  char **v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // [rsp-50h] [rbp-50h]
  unsigned __int64 v42; // [rsp-48h] [rbp-48h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  unsigned __int64 v44; // [rsp-40h] [rbp-40h]
  unsigned __int64 v45; // [rsp-40h] [rbp-40h]
  char **v46; // [rsp-40h] [rbp-40h]
  char **v47; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v43 = a2 + 16;
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v12 = v11;
      if ( v11 <= v9 )
      {
        v26 = *(_QWORD *)a1;
        if ( v11 )
        {
          v38 = v8 + 8;
          v39 = (char **)(a2 + 24);
          do
          {
            v40 = v38;
            v47 = v39;
            v38 += 88;
            *(_QWORD *)(v38 - 96) = *(v39 - 1);
            sub_23036B0(v40, v39, (__int64)v39, a4, v10, a6);
            v39 = v47 + 11;
          }
          while ( v8 + 8 + 88 * v11 != v38 );
          v26 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = v8 + 88 * v11;
        }
        v27 = v26 + 88 * v9;
        while ( v10 != v27 )
        {
          v27 -= 88LL;
          v28 = *(_QWORD *)(v27 + 8);
          if ( v28 != v27 + 24 )
          {
            v45 = v10;
            _libc_free(v28);
            v10 = v45;
          }
        }
        *(_DWORD *)(a1 + 8) = v11;
        v29 = *(_QWORD *)a2;
        v30 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v30 )
        {
          do
          {
            v30 -= 88;
            v31 = *(_QWORD *)(v30 + 8);
            if ( v31 != v30 + 24 )
              _libc_free(v31);
          }
          while ( v29 != v30 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v35 = 5 * v9;
          v36 = v8 + 88 * v9;
          while ( v36 != v8 )
          {
            while ( 1 )
            {
              v36 -= 88LL;
              v37 = *(_QWORD *)(v36 + 8);
              if ( v37 == v36 + 24 )
                break;
              _libc_free(v37);
              if ( v36 == v8 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          v9 = 0;
          sub_23590C0(a1, v11, v35, a4, v10, a6);
          v11 = *(unsigned int *)(a2 + 8);
          v8 = *(_QWORD *)a1;
          v43 = *(_QWORD *)a2;
          v13 = *(_QWORD *)a2;
        }
        else
        {
          v13 = v43;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v32 = v8 + 8;
            v33 = (char **)(a2 + 24);
            v41 = 88 * v9;
            v9 *= 88LL;
            v42 = v32 + v9;
            do
            {
              v34 = v32;
              v46 = v33;
              v32 += 88;
              *(_QWORD *)(v32 - 96) = *(v33 - 1);
              sub_23036B0(v34, v33, (__int64)v33, a4, v10, a6);
              v33 = v46 + 11;
            }
            while ( v32 != v42 );
            v11 = *(unsigned int *)(a2 + 8);
            v8 = *(_QWORD *)a1;
            v43 = *(_QWORD *)a2;
            v13 = *(_QWORD *)a2 + v41;
          }
        }
        v14 = v43;
        v15 = v8 + v9;
        v16 = v13;
        v17 = v43 + 88 * v11;
        if ( v17 != v13 )
        {
          do
          {
            while ( 1 )
            {
              if ( v15 )
              {
                v18 = *(_QWORD *)v16;
                *(_DWORD *)(v15 + 16) = 0;
                *(_DWORD *)(v15 + 20) = 8;
                *(_QWORD *)v15 = v18;
                *(_QWORD *)(v15 + 8) = v15 + 24;
                v19 = *(unsigned int *)(v16 + 16);
                if ( (_DWORD)v19 )
                  break;
              }
              v16 += 88;
              v15 += 88LL;
              if ( v17 == v16 )
                goto LABEL_12;
            }
            v20 = (char **)(v16 + 8);
            v44 = v17;
            v16 += 88;
            sub_23036B0(v15 + 8, v20, v19, v14, v10, a6);
            v17 = v44;
            v15 += 88LL;
          }
          while ( v44 != v16 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v12;
        v21 = *(_QWORD *)a2;
        v22 = *(_QWORD *)a2 + 88LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v22 )
        {
          do
          {
            v22 -= 88;
            v23 = *(_QWORD *)(v22 + 8);
            if ( v23 != v22 + 24 )
              _libc_free(v23);
          }
          while ( v21 != v22 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v24 = v8 + 88 * v9;
      if ( v24 != v8 )
      {
        do
        {
          v24 -= 88LL;
          v25 = *(_QWORD *)(v24 + 8);
          if ( v25 != v24 + 24 )
            _libc_free(v25);
        }
        while ( v24 != v8 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
        _libc_free(v10);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v43;
    }
  }
}
