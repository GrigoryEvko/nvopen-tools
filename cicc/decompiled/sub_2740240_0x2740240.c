// Function: sub_2740240
// Address: 0x2740240
//
void __fastcall sub_2740240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // r14
  unsigned __int64 v11; // r15
  unsigned __int64 v12; // rdx
  char **v13; // rbx
  char **v14; // r15
  char **v15; // rsi
  __int64 v16; // rdi
  char **v17; // r13
  unsigned __int64 *v18; // rbx
  unsigned __int64 *v19; // rbx
  __int64 v20; // rdx
  unsigned __int64 *v21; // rbx
  char **v22; // r13
  unsigned __int64 *v23; // rbx
  __int64 v24; // rbx
  unsigned __int64 v25; // r15
  __int64 v26; // rdi
  unsigned __int64 *v27; // rbx
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  int v32; // r15d
  unsigned __int64 v33; // rbx
  __int64 v34; // rdi
  int v35; // [rsp-54h] [rbp-54h]
  char **v36; // [rsp-50h] [rbp-50h]
  char **v37; // [rsp-50h] [rbp-50h]
  char **v38; // [rsp-50h] [rbp-50h]
  char **v39; // [rsp-50h] [rbp-50h]
  unsigned __int64 v40; // [rsp-40h] [rbp-40h] BYREF

  if ( a1 != a2 )
  {
    v6 = (char **)(a2 + 16);
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v35 = *(_DWORD *)(a2 + 8);
      if ( v11 <= v9 )
      {
        v20 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v33 = v10 + 80 * v11;
          do
          {
            v34 = v10;
            v39 = v6;
            v10 += 80;
            sub_2738790(v34, v6, v20, v9, a5, a6);
            v6 = v39 + 10;
          }
          while ( v10 != v33 );
          v20 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v21 = (unsigned __int64 *)(v20 + 80 * v9);
        while ( (unsigned __int64 *)v10 != v21 )
        {
          v21 -= 10;
          if ( (unsigned __int64 *)*v21 != v21 + 2 )
            _libc_free(*v21);
        }
        *(_DWORD *)(a1 + 8) = v35;
        v22 = *(char ***)a2;
        v23 = (unsigned __int64 *)(*(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v23 )
        {
          do
          {
            v23 -= 10;
            if ( (unsigned __int64 *)*v23 != v23 + 2 )
              _libc_free(*v23);
          }
          while ( v22 != (char **)v23 );
        }
      }
      else
      {
        v12 = *(unsigned int *)(a1 + 12);
        if ( v11 > v12 )
        {
          v27 = (unsigned __int64 *)(v10 + 80 * v9);
          while ( (unsigned __int64 *)v10 != v27 )
          {
            while ( 1 )
            {
              v27 -= 10;
              if ( (unsigned __int64 *)*v27 == v27 + 2 )
                break;
              _libc_free(*v27);
              if ( (unsigned __int64 *)v10 == v27 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          v10 = sub_C8D7D0(a1, a1 + 16, v11, 0x50u, &v40, a6);
          sub_2740170(a1, v10, v28, v29, v30, v31);
          v32 = v40;
          if ( a1 + 16 != *(_QWORD *)a1 )
            _libc_free(*(_QWORD *)a1);
          *(_QWORD *)a1 = v10;
          *(_DWORD *)(a1 + 12) = v32;
          v6 = *(char ***)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v13 = *(char ***)a2;
        }
        else
        {
          v13 = (char **)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v24 = 80 * v9;
            v25 = v10 + 80 * v9;
            do
            {
              v26 = v10;
              v38 = v6;
              v10 += 80;
              sub_2738790(v26, v6, v12, v9, a5, a6);
              v6 = v38 + 10;
            }
            while ( v25 != v10 );
            v6 = *(char ***)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = v24 + *(_QWORD *)a1;
            v13 = (char **)(*(_QWORD *)a2 + v24);
          }
        }
        v14 = &v6[10 * v11];
        while ( v14 != v13 )
        {
          while ( 1 )
          {
            if ( v10 )
            {
              *(_DWORD *)(v10 + 8) = 0;
              *(_QWORD *)v10 = v10 + 16;
              *(_DWORD *)(v10 + 12) = 8;
              if ( *((_DWORD *)v13 + 2) )
                break;
            }
            v13 += 10;
            v10 += 80;
            if ( v14 == v13 )
              goto LABEL_12;
          }
          v15 = v13;
          v16 = v10;
          v13 += 10;
          v10 += 80;
          sub_2738790(v16, v15, v12, v9, a5, a6);
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v35;
        v17 = *(char ***)a2;
        v18 = (unsigned __int64 *)(*(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v18 )
        {
          do
          {
            v18 -= 10;
            if ( (unsigned __int64 *)*v18 != v18 + 2 )
              _libc_free(*v18);
          }
          while ( v17 != (char **)v18 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v19 = (unsigned __int64 *)(v10 + 80 * v9);
      if ( v19 != (unsigned __int64 *)v10 )
      {
        do
        {
          v19 -= 10;
          if ( (unsigned __int64 *)*v19 != v19 + 2 )
          {
            v36 = v6;
            _libc_free(*v19);
            v6 = v36;
          }
        }
        while ( v19 != (unsigned __int64 *)v10 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
      {
        v37 = v6;
        _libc_free(v10);
        v6 = v37;
      }
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
