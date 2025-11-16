// Function: sub_1995960
// Address: 0x1995960
//
void __fastcall sub_1995960(__int64 a1, __int64 a2)
{
  _QWORD *v3; // r12
  _QWORD *v4; // r8
  unsigned __int64 v5; // r9
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // r15
  _QWORD *v8; // rax
  __int64 v9; // rdx
  _QWORD *v10; // rbx
  _QWORD *v11; // r15
  _QWORD *v12; // r12
  _QWORD *v13; // r12
  __int64 v14; // rbx
  unsigned __int64 v15; // rdi
  _QWORD *v16; // r15
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rax
  _QWORD *v19; // rbx
  unsigned __int64 v20; // rdi
  _QWORD *v21; // r12
  __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  __int64 v24; // rbx
  _QWORD *v25; // r15
  __int64 v26; // rax
  _QWORD *v27; // r12
  unsigned __int64 v28; // rdi
  _QWORD *v29; // rbx
  __int64 v30; // rax
  unsigned __int64 v31; // [rsp-50h] [rbp-50h]
  int v32; // [rsp-44h] [rbp-44h]
  _QWORD *v33; // [rsp-40h] [rbp-40h]
  _QWORD *v34; // [rsp-40h] [rbp-40h]
  _QWORD *v35; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v3 = (_QWORD *)(a2 + 16);
    v4 = *(_QWORD **)a1;
    v5 = *(unsigned int *)(a1 + 8);
    v6 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v7 = *(unsigned int *)(a2 + 8);
      v32 = *(_DWORD *)(a2 + 8);
      if ( v7 <= v5 )
      {
        v18 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v29 = &v4[10 * v7];
          do
          {
            *v4 = *v3;
            v4[1] = v3[1];
            if ( v4 != v3 )
            {
              v35 = v4;
              sub_16CCF00((__int64)(v4 + 2), 2, (__int64)(v3 + 2));
              v4 = v35;
            }
            v30 = v3[9];
            v4 += 10;
            v3 += 10;
            *(v4 - 1) = v30;
          }
          while ( v4 != v29 );
          v18 = *(_QWORD *)a1;
          v5 = *(unsigned int *)(a1 + 8);
        }
        v19 = (_QWORD *)(v18 + 80 * v5);
        while ( v4 != v19 )
        {
          v19 -= 10;
          v20 = v19[4];
          if ( v20 != v19[3] )
          {
            v33 = v4;
            _libc_free(v20);
            v4 = v33;
          }
        }
        *(_DWORD *)(a1 + 8) = v32;
        v21 = *(_QWORD **)a2;
        v22 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v22 )
        {
          do
          {
            v22 -= 80;
            v23 = *(_QWORD *)(v22 + 32);
            if ( v23 != *(_QWORD *)(v22 + 24) )
              _libc_free(v23);
          }
          while ( v21 != (_QWORD *)v22 );
        }
      }
      else
      {
        if ( v7 > *(unsigned int *)(a1 + 12) )
        {
          v27 = &v4[10 * v5];
          while ( v27 != (_QWORD *)v6 )
          {
            while ( 1 )
            {
              v27 -= 10;
              v28 = v27[4];
              if ( v28 == v27[3] )
                break;
              _libc_free(v28);
              if ( v27 == (_QWORD *)v6 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          sub_19957D0((unsigned __int64 *)a1, v7);
          v3 = *(_QWORD **)a2;
          v7 = *(unsigned int *)(a2 + 8);
          v5 = 0;
          v6 = *(_QWORD *)a1;
          v8 = *(_QWORD **)a2;
        }
        else
        {
          v8 = (_QWORD *)(a2 + 16);
          if ( *(_DWORD *)(a1 + 8) )
          {
            v24 = 80 * v5;
            v5 = v24;
            v25 = &v4[(unsigned __int64)v24 / 8];
            do
            {
              *v4 = *v3;
              v4[1] = v3[1];
              if ( v4 != v3 )
              {
                v31 = v5;
                v34 = v4;
                sub_16CCF00((__int64)(v4 + 2), 2, (__int64)(v3 + 2));
                v5 = v31;
                v4 = v34;
              }
              v26 = v3[9];
              v4 += 10;
              v3 += 10;
              *(v4 - 1) = v26;
            }
            while ( v25 != v4 );
            v3 = *(_QWORD **)a2;
            v7 = *(unsigned int *)(a2 + 8);
            v8 = (_QWORD *)(*(_QWORD *)a2 + v24);
            v6 = *(_QWORD *)a1;
          }
        }
        v9 = 5 * v7;
        v10 = (_QWORD *)(v5 + v6);
        v11 = v8;
        v12 = &v3[2 * v9];
        if ( v12 != v8 )
        {
          do
          {
            if ( v10 )
            {
              *v10 = *v11;
              v10[1] = v11[1];
              sub_16CCEE0(v10 + 2, (__int64)(v10 + 7), 2, (__int64)(v11 + 2));
              v10[9] = v11[9];
            }
            v11 += 10;
            v10 += 10;
          }
          while ( v12 != v11 );
        }
        *(_DWORD *)(a1 + 8) = v32;
        v13 = *(_QWORD **)a2;
        v14 = *(_QWORD *)a2 + 80LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v14 )
        {
          do
          {
            v14 -= 80;
            v15 = *(_QWORD *)(v14 + 32);
            if ( v15 != *(_QWORD *)(v14 + 24) )
              _libc_free(v15);
          }
          while ( v13 != (_QWORD *)v14 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v16 = &v4[10 * v5];
      if ( v16 != v4 )
      {
        do
        {
          v16 -= 10;
          v17 = v16[4];
          if ( v17 != v16[3] )
            _libc_free(v17);
        }
        while ( v16 != (_QWORD *)v6 );
        v4 = *(_QWORD **)a1;
      }
      if ( v4 != (_QWORD *)(a1 + 16) )
        _libc_free((unsigned __int64)v4);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v3;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
