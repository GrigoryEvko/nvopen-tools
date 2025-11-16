// Function: sub_18E8A60
// Address: 0x18e8a60
//
void __fastcall sub_18E8A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, int a6)
{
  char **v6; // r15
  unsigned __int64 v8; // r8
  unsigned __int64 v9; // rdx
  unsigned __int64 *v10; // rbx
  unsigned __int64 v11; // rsi
  int v12; // r14d
  __int64 v13; // rcx
  __int64 v14; // rbx
  char **v15; // rax
  __int64 i; // r15
  __int64 v17; // rdx
  unsigned __int64 *v18; // r13
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r14
  __int64 v21; // rcx
  unsigned __int64 *v22; // rbx
  unsigned __int64 *v23; // r13
  unsigned __int64 *v24; // rbx
  __int64 v25; // rcx
  unsigned __int64 v26; // rbx
  char **v27; // rsi
  unsigned __int64 *v28; // r15
  __int64 v29; // rdx
  unsigned __int64 v30; // rbx
  char **v31; // rsi
  __int64 v32; // [rsp-50h] [rbp-50h]
  unsigned __int64 v33; // [rsp-48h] [rbp-48h]
  char **v34; // [rsp-40h] [rbp-40h]
  unsigned __int64 v35; // [rsp-40h] [rbp-40h]
  unsigned __int64 v36; // [rsp-40h] [rbp-40h]
  unsigned __int64 v37; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = (char **)(a2 + 16);
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v10 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v12 = v11;
      if ( v11 <= v9 )
      {
        v21 = *(_QWORD *)a1;
        if ( v11 )
        {
          v29 = 19 * v11;
          v30 = v8 + 152 * v11;
          do
          {
            v31 = v6;
            v37 = v8;
            v6 += 19;
            sub_18E63F0(v8, v31, v29, v21, v8, a6);
            v29 = (__int64)*(v6 - 1);
            v8 = v37 + 152;
            *(_QWORD *)(v37 + 144) = v29;
          }
          while ( v37 + 152 != v30 );
          v21 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
        }
        v22 = (unsigned __int64 *)(v21 + 152 * v9);
        while ( (unsigned __int64 *)v8 != v22 )
        {
          v22 -= 19;
          if ( (unsigned __int64 *)*v22 != v22 + 2 )
          {
            v35 = v8;
            _libc_free(*v22);
            v8 = v35;
          }
        }
        *(_DWORD *)(a1 + 8) = v12;
        v23 = *(unsigned __int64 **)a2;
        v24 = (unsigned __int64 *)(*(_QWORD *)a2 + 152LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v24 )
        {
          do
          {
            v24 -= 19;
            if ( (unsigned __int64 *)*v24 != v24 + 2 )
              _libc_free(*v24);
          }
          while ( v23 != v24 );
        }
      }
      else
      {
        if ( v11 > *(unsigned int *)(a1 + 12) )
        {
          v28 = (unsigned __int64 *)(v8 + 152 * v9);
          while ( v28 != v10 )
          {
            while ( 1 )
            {
              v28 -= 19;
              if ( (unsigned __int64 *)*v28 == v28 + 2 )
                break;
              _libc_free(*v28);
              if ( v28 == v10 )
                goto LABEL_44;
            }
          }
LABEL_44:
          *(_DWORD *)(a1 + 8) = 0;
          sub_18E88A0(a1, v11);
          v6 = *(char ***)a2;
          v11 = *(unsigned int *)(a2 + 8);
          v9 = 0;
          v10 = *(unsigned __int64 **)a1;
          v13 = *(_QWORD *)a2;
        }
        else
        {
          v13 = (__int64)v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v25 = 9 * v9;
            v32 = 152 * v9;
            v9 *= 152LL;
            v26 = v8 + v9;
            do
            {
              v27 = v6;
              v33 = v9;
              v6 += 19;
              v36 = v8;
              sub_18E63F0(v8, v27, v9, v25, v8, a6);
              v25 = (__int64)*(v6 - 1);
              v9 = v33;
              *(_QWORD *)(v36 + 144) = v25;
              v8 = v36 + 152;
            }
            while ( v26 != v36 + 152 );
            v6 = *(char ***)a2;
            v11 = *(unsigned int *)(a2 + 8);
            v10 = *(unsigned __int64 **)a1;
            v13 = *(_QWORD *)a2 + v32;
          }
        }
        v14 = (__int64)v10 + v9;
        v15 = &v6[19 * v11];
        for ( i = v13; v15 != (char **)i; v14 += 152 )
        {
          if ( v14 )
          {
            *(_DWORD *)(v14 + 8) = 0;
            *(_QWORD *)v14 = v14 + 16;
            *(_DWORD *)(v14 + 12) = 8;
            v17 = *(unsigned int *)(i + 8);
            if ( (_DWORD)v17 )
            {
              v34 = v15;
              sub_18E63F0(v14, (char **)i, v17, v13, v8, a6);
              v15 = v34;
            }
            *(_QWORD *)(v14 + 144) = *(_QWORD *)(i + 144);
          }
          i += 152;
        }
        *(_DWORD *)(a1 + 8) = v12;
        v18 = *(unsigned __int64 **)a2;
        v19 = (unsigned __int64 *)(*(_QWORD *)a2 + 152LL * *(unsigned int *)(a2 + 8));
        if ( *(unsigned __int64 **)a2 != v19 )
        {
          do
          {
            v19 -= 19;
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
      v20 = (unsigned __int64 *)(v8 + 152 * v9);
      if ( v20 != (unsigned __int64 *)v8 )
      {
        do
        {
          v20 -= 19;
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
