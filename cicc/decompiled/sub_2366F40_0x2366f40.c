// Function: sub_2366F40
// Address: 0x2366f40
//
void __fastcall sub_2366F40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char **v6; // r15
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // r13
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // rbx
  __int64 v15; // r13
  char **v16; // rax
  __int64 v17; // rdx
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r13
  __int64 v21; // rdx
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // rbx
  __int64 v25; // rbx
  char **v26; // rsi
  unsigned __int64 *v27; // r13
  unsigned __int64 v28; // rbx
  char **v29; // rsi
  int v30; // [rsp-44h] [rbp-44h]
  char **v31; // [rsp-40h] [rbp-40h]
  unsigned __int64 v32; // [rsp-40h] [rbp-40h]
  unsigned __int64 v33; // [rsp-40h] [rbp-40h]
  unsigned __int64 v34; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = (char **)(a2 + 16);
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v30 = v11;
      if ( v11 <= v9 )
      {
        v21 = *(_QWORD *)a1;
        if ( v11 )
        {
          v28 = v8 + 160 * v11;
          do
          {
            v29 = v6;
            v34 = v8;
            v6 += 20;
            sub_2303A20(v8, v29, v21, a4, v8, a6);
            *(_QWORD *)(v34 + 144) = *(v6 - 2);
            v21 = (__int64)*(v6 - 1);
            v8 = v34 + 160;
            *(_QWORD *)(v34 + 152) = v21;
          }
          while ( v34 + 160 != v28 );
          v21 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v22 = (unsigned __int64 *)(v21 + 160 * v9);
        while ( (unsigned __int64 *)v8 != v22 )
        {
          v22 -= 20;
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
          {
            v32 = v8;
            _libc_free(*v22);
            v8 = v32;
          }
        }
        *(_DWORD *)(a1 + 8) = v30;
        v23 = *(unsigned __int64 **)a2;
        v24 = (unsigned __int64 *)(*(_QWORD *)a2 + 160LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v24 )
        {
          do
          {
            v24 -= 20;
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              _libc_free(*v24);
          }
          while ( v23 != v24 );
        }
      }
      else
      {
        v12 = *(unsigned int *)(a1 + 12);
        if ( v11 > v12 )
        {
          v27 = (unsigned __int64 *)(v8 + 160 * v9);
          while ( v27 != v10 )
          {
            while ( 1 )
            {
              v27 -= 20;
              if ( (unsigned __int64 *)*v27 == v27 + 2 )
                break;
              _libc_free(*v27);
              if ( v27 == v10 )
                goto LABEL_44;
            }
          }
LABEL_44:
          *(_DWORD *)(a1 + 8) = 0;
          v9 = 0;
          sub_2366E20(a1, v11, v12, a4, v8, a6);
          v6 = *(char ***)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v10 = *(unsigned __int64 **)a1;
          v13 = *(_QWORD *)a2;
        }
        else
        {
          v13 = (__int64)v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v9 *= 160LL;
            v25 = v8 + v9;
            do
            {
              v26 = v6;
              v33 = v8;
              v6 += 20;
              sub_2303A20(v8, v26, v12, v13, v8, a6);
              *(_QWORD *)(v33 + 144) = *(v6 - 2);
              v12 = (unsigned __int64)*(v6 - 1);
              v8 = v33 + 160;
              *(_QWORD *)(v33 + 152) = v12;
            }
            while ( v25 != v33 + 160 );
            v6 = *(char ***)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = *(unsigned __int64 **)a1;
            v13 = *(_QWORD *)a2 + v9;
          }
        }
        v14 = (__int64)v10 + v9;
        v15 = v13;
        v16 = &v6[20 * v11];
        if ( v16 != (char **)v13 )
        {
          do
          {
            if ( v14 )
            {
              *(_DWORD *)(v14 + 8) = 0;
              *(_QWORD *)v14 = v14 + 16;
              *(_DWORD *)(v14 + 12) = 8;
              v17 = *(unsigned int *)(v15 + 8);
              if ( (_DWORD)v17 )
              {
                v31 = v16;
                sub_2303A20(v14, (char **)v15, v17, v13, v8, a6);
                v16 = v31;
              }
              *(_QWORD *)(v14 + 144) = *(_QWORD *)(v15 + 144);
              *(_QWORD *)(v14 + 152) = *(_QWORD *)(v15 + 152);
            }
            v15 += 160;
            v14 += 160;
          }
          while ( v16 != (char **)v15 );
        }
        *(_DWORD *)(a1 + 8) = v30;
        v18 = *(unsigned __int64 **)a2;
        v19 = (unsigned __int64 *)(*(_QWORD *)a2 + 160LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v19 )
        {
          do
          {
            v19 -= 20;
            if ( (unsigned __int64 *)*v19 != v19 + 2 )
              _libc_free(*v19);
          }
          while ( v18 != v19 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v20 = (unsigned __int64 *)(v8 + 160 * v9);
      if ( v20 != (unsigned __int64 *)v8 )
      {
        do
        {
          v20 -= 20;
          if ( (unsigned __int64 *)*v20 != v20 + 2 )
            _libc_free(*v20);
        }
        while ( v20 != v10 );
        v8 = *(_QWORD *)a1;
      }
      if ( v8 != a1 + 16 )
        _libc_free(v8);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v6;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
