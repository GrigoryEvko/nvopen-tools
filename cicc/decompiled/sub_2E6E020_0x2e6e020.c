// Function: sub_2E6E020
// Address: 0x2e6e020
//
void __fastcall sub_2E6E020(__int64 a1, __int64 a2)
{
  unsigned __int64 *v2; // rcx
  unsigned __int64 v5; // rbx
  unsigned __int64 v6; // r15
  char *v7; // rdx
  char *v8; // rcx
  _QWORD *v9; // rax
  _QWORD *v10; // rsi
  __int64 v11; // r15
  unsigned __int64 v12; // r12
  unsigned __int64 v13; // rbx
  unsigned __int64 v14; // rdi
  unsigned __int64 *v15; // r12
  unsigned __int64 *v16; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 v18; // r15
  unsigned __int64 v19; // rdi
  unsigned __int64 *v20; // r15
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // r12
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int64 *v28; // rbx
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r15
  unsigned __int64 v31; // rdi
  unsigned __int64 *v32; // [rsp-50h] [rbp-50h]
  unsigned __int64 *v33; // [rsp-50h] [rbp-50h]
  unsigned __int64 *v34; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v35; // [rsp-48h] [rbp-48h]
  unsigned __int64 *v36; // [rsp-48h] [rbp-48h]
  int v37; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v38; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v39; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v40; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v2 = (unsigned __int64 *)(a2 + 16);
    v5 = *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v6 = *(unsigned int *)(a2 + 8);
      v37 = *(_DWORD *)(a2 + 8);
      if ( v6 <= v5 )
      {
        v15 = *(unsigned __int64 **)a1;
        v16 = *(unsigned __int64 **)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v28 = &v15[v6];
          do
          {
            v29 = *v2;
            *v2 = 0;
            v30 = *v15;
            *v15 = v29;
            if ( v30 )
            {
              v31 = *(_QWORD *)(v30 + 24);
              if ( v31 != v30 + 40 )
              {
                v35 = v2;
                _libc_free(v31);
                v2 = v35;
              }
              v36 = v2;
              j_j___libc_free_0(v30);
              v2 = v36;
            }
            ++v15;
            ++v2;
          }
          while ( v15 != v28 );
          v16 = *(unsigned __int64 **)a1;
          v5 = *(unsigned int *)(a1 + 8);
        }
        for ( i = &v16[v5]; v15 != i; --i )
        {
          v18 = *(i - 1);
          if ( v18 )
          {
            v19 = *(_QWORD *)(v18 + 24);
            if ( v19 != v18 + 40 )
              _libc_free(v19);
            j_j___libc_free_0(v18);
          }
        }
      }
      else
      {
        if ( v6 > *(unsigned int *)(a1 + 12) )
        {
          sub_2E6DCE0((__int64 *)a1);
          v5 = 0;
          sub_239B9C0(a1, v6, v24, v25, v26, v27);
          v2 = *(unsigned __int64 **)a2;
          v6 = *(unsigned int *)(a2 + 8);
          v7 = *(char **)a2;
        }
        else
        {
          v7 = (char *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v20 = *(unsigned __int64 **)a1;
            v5 *= 8LL;
            v34 = (unsigned __int64 *)(*(_QWORD *)a1 + v5);
            do
            {
              v21 = *v2;
              *v2 = 0;
              v22 = *v20;
              *v20 = v21;
              if ( v22 )
              {
                v23 = *(_QWORD *)(v22 + 24);
                if ( v23 != v22 + 40 )
                {
                  v32 = v2;
                  _libc_free(v23);
                  v2 = v32;
                }
                v33 = v2;
                j_j___libc_free_0(v22);
                v2 = v33;
              }
              ++v2;
              ++v20;
            }
            while ( v20 != v34 );
            v2 = *(unsigned __int64 **)a2;
            v6 = *(unsigned int *)(a2 + 8);
            v7 = (char *)(*(_QWORD *)a2 + v5);
          }
        }
        v8 = (char *)&v2[v6];
        v9 = (_QWORD *)(v5 + *(_QWORD *)a1);
        v10 = (_QWORD *)((char *)v9 + v8 - v7);
        if ( v8 != v7 )
        {
          do
          {
            if ( v9 )
            {
              *v9 = *(_QWORD *)v7;
              *(_QWORD *)v7 = 0;
            }
            ++v9;
            v7 += 8;
          }
          while ( v9 != v10 );
        }
      }
      *(_DWORD *)(a1 + 8) = v37;
      sub_2E6DCE0((__int64 *)a2);
    }
    else
    {
      v11 = *(_QWORD *)a1;
      v12 = *(_QWORD *)a1 + 8 * v5;
      if ( *(_QWORD *)a1 != v12 )
      {
        do
        {
          v13 = *(_QWORD *)(v12 - 8);
          v12 -= 8LL;
          if ( v13 )
          {
            v14 = *(_QWORD *)(v13 + 24);
            if ( v14 != v13 + 40 )
            {
              v38 = v2;
              _libc_free(v14);
              v2 = v38;
            }
            v39 = v2;
            j_j___libc_free_0(v13);
            v2 = v39;
          }
        }
        while ( v11 != v12 );
        v12 = *(_QWORD *)a1;
      }
      if ( v12 != a1 + 16 )
      {
        v40 = v2;
        _libc_free(v12);
        v2 = v40;
      }
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v2;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
