// Function: sub_F312A0
// Address: 0xf312a0
//
void __fastcall sub_F312A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r14
  unsigned __int64 v9; // rbx
  __int64 v10; // r8
  unsigned __int64 v11; // rsi
  int v12; // r15d
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  unsigned __int64 v15; // rbx
  __int64 v16; // r14
  unsigned __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // r13
  __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned __int64 v23; // r15
  __int64 v24; // rdi
  __int64 v25; // rdx
  unsigned __int64 v26; // rbx
  __int64 v27; // rdi
  __int64 v28; // r13
  __int64 v29; // rbx
  __int64 v30; // rdi
  char **v31; // r14
  char **v32; // rsi
  unsigned __int64 v33; // rbx
  __int64 v34; // rdi
  char **v35; // rbx
  __int64 v36; // rdx
  char **v37; // rsi
  __int64 v38; // [rsp-50h] [rbp-50h]
  __int64 v39; // [rsp-48h] [rbp-48h]
  __int64 v40; // [rsp-40h] [rbp-40h]
  unsigned __int64 v41; // [rsp-40h] [rbp-40h]
  __int64 v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]
  __int64 v44; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v40 = a2 + 16;
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v12 = v11;
      if ( v11 <= v9 )
      {
        v25 = *(_QWORD *)a1;
        if ( v11 )
        {
          v35 = (char **)(a2 + 24);
          v36 = v8 + 8;
          v38 = 80 * v11;
          v39 = a2 + 24 + 80 * v11;
          do
          {
            v44 = v36;
            *(_QWORD *)(v36 - 8) = *(v35 - 1);
            v37 = v35;
            v35 += 10;
            sub_F2FA90(v36, v37, v36, a4, v10, a6);
            v11 = *((unsigned int *)v35 - 4);
            v36 = v44 + 80;
            *(_DWORD *)(v44 + 64) = v11;
          }
          while ( (char **)v39 != v35 );
          v25 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = v8 + v38;
        }
        v26 = v25 + 80 * v9;
        while ( v10 != v26 )
        {
          v26 -= 80LL;
          v27 = *(_QWORD *)(v26 + 8);
          if ( v27 != v26 + 24 )
          {
            v42 = v10;
            _libc_free(v27, v11);
            v10 = v42;
          }
        }
        *(_DWORD *)(a1 + 8) = v12;
        v28 = *(_QWORD *)a2;
        v29 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v29 )
        {
          do
          {
            v29 -= 80;
            v30 = *(_QWORD *)(v29 + 8);
            if ( v30 != v29 + 24 )
              _libc_free(v30, v11);
          }
          while ( v28 != v29 );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v11 > v13 )
        {
          v33 = v8 + 80 * v9;
          while ( v33 != v8 )
          {
            while ( 1 )
            {
              v33 -= 80LL;
              v34 = *(_QWORD *)(v33 + 8);
              if ( v34 == v33 + 24 )
                break;
              _libc_free(v34, v11);
              if ( v33 == v8 )
                goto LABEL_44;
            }
          }
LABEL_44:
          *(_DWORD *)(a1 + 8) = 0;
          v9 = 0;
          sub_F30F50(a1, v11, v13, a4, v10, a6);
          v11 = *(unsigned int *)(a2 + 8);
          v8 = *(_QWORD *)a1;
          v40 = *(_QWORD *)a2;
          v14 = *(_QWORD *)a2;
        }
        else
        {
          v14 = v40;
          if ( *(_DWORD *)(a1 + 8) )
          {
            a4 = v8 + 8;
            v9 *= 80LL;
            v31 = (char **)(a2 + 24);
            do
            {
              v43 = a4;
              *(_QWORD *)(a4 - 8) = *(v31 - 1);
              v32 = v31;
              v31 += 10;
              sub_F2FA90(a4, v32, v14, a4, v10, a6);
              a4 = v43 + 80;
              *(_DWORD *)(v43 + 64) = *((_DWORD *)v31 - 4);
            }
            while ( v31 != (char **)(a2 + 24 + v9) );
            v11 = *(unsigned int *)(a2 + 8);
            v8 = *(_QWORD *)a1;
            v40 = *(_QWORD *)a2;
            v14 = *(_QWORD *)a2 + v9;
          }
        }
        v15 = v8 + v9;
        v16 = v14;
        v17 = v40 + 80 * v11;
        if ( v17 != v14 )
        {
          do
          {
            if ( v15 )
            {
              v18 = *(_QWORD *)v16;
              *(_DWORD *)(v15 + 16) = 0;
              *(_DWORD *)(v15 + 20) = 6;
              *(_QWORD *)v15 = v18;
              *(_QWORD *)(v15 + 8) = v15 + 24;
              v19 = *(unsigned int *)(v16 + 16);
              if ( (_DWORD)v19 )
              {
                v11 = v16 + 8;
                v41 = v17;
                sub_F2FA90(v15 + 8, (char **)(v16 + 8), v19, a4, v10, a6);
                v17 = v41;
              }
              *(_DWORD *)(v15 + 72) = *(_DWORD *)(v16 + 72);
            }
            v16 += 80;
            v15 += 80LL;
          }
          while ( v17 != v16 );
        }
        *(_DWORD *)(a1 + 8) = v12;
        v20 = *(_QWORD *)a2;
        v21 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v21 -= 80;
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 != v21 + 24 )
              _libc_free(v22, v11);
          }
          while ( v20 != v21 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v23 = v8 + 80 * v9;
      if ( v23 != v8 )
      {
        do
        {
          v23 -= 80LL;
          v24 = *(_QWORD *)(v23 + 8);
          if ( v24 != v23 + 24 )
            _libc_free(v24, a2);
        }
        while ( v23 != v8 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
        _libc_free(v10, a2);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v40;
    }
  }
}
