// Function: sub_239AF40
// Address: 0x239af40
//
void __fastcall sub_239AF40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r15
  unsigned __int64 *v7; // r9
  unsigned __int64 v8; // rbx
  unsigned __int64 *v9; // r10
  unsigned __int64 v10; // rsi
  _QWORD *v11; // r13
  char *v12; // rbx
  _QWORD *i; // r8
  __int64 v14; // rdx
  _QWORD *v15; // r13
  _QWORD *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 *v18; // rbx
  __int64 v19; // rdx
  unsigned __int64 *v20; // rdx
  unsigned __int64 *v21; // rbx
  _QWORD *v22; // r13
  _QWORD *v23; // rbx
  unsigned __int64 *v24; // r13
  unsigned __int64 v25; // rdx
  unsigned __int64 v26; // rax
  int v27; // eax
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // rbx
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  int v32; // eax
  int v33; // [rsp-4Ch] [rbp-4Ch]
  unsigned __int64 *v34; // [rsp-48h] [rbp-48h]
  _QWORD *v35; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v36; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v37; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v38; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v39; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v40; // [rsp-40h] [rbp-40h]
  unsigned __int64 *v41; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v5 = (_QWORD *)(a2 + 16);
    v7 = *(unsigned __int64 **)a1;
    v8 = *(unsigned int *)(a1 + 8);
    v9 = *(unsigned __int64 **)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v10 = *(unsigned int *)(a2 + 8);
      v33 = v10;
      if ( v10 <= v8 )
      {
        v20 = *(unsigned __int64 **)a1;
        if ( v10 )
        {
          v29 = &v7[4 * v10];
          do
          {
            v30 = v7[2];
            v31 = v5[2];
            if ( v30 != v31 )
            {
              if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
              {
                v40 = v7;
                sub_BD60C0(v7);
                v31 = v5[2];
                v7 = v40;
              }
              v7[2] = v31;
              if ( v31 != -4096 && v31 != 0 && v31 != -8192 )
              {
                v41 = v7;
                sub_BD6050(v7, *v5 & 0xFFFFFFFFFFFFFFF8LL);
                v7 = v41;
              }
            }
            v32 = *((_DWORD *)v5 + 6);
            v7 += 4;
            v5 += 4;
            *((_DWORD *)v7 - 2) = v32;
          }
          while ( v7 != v29 );
          v20 = *(unsigned __int64 **)a1;
          v8 = *(unsigned int *)(a1 + 8);
        }
        v21 = &v20[4 * v8];
        if ( v21 != v7 )
        {
          do
          {
            v21 -= 4;
            v37 = v7;
            sub_D68D70(v21);
            v7 = v37;
          }
          while ( v37 != v21 );
        }
        *(_DWORD *)(a1 + 8) = v10;
        v22 = *(_QWORD **)a2;
        v23 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v23 )
        {
          do
          {
            v23 -= 4;
            sub_D68D70(v23);
          }
          while ( v22 != v23 );
        }
      }
      else
      {
        if ( v10 > *(unsigned int *)(a1 + 12) )
        {
          v28 = &v7[4 * v8];
          if ( v28 != v7 )
          {
            do
            {
              v28 -= 4;
              v34 = v9;
              sub_D68D70(v28);
              v9 = v34;
            }
            while ( v28 != v34 );
          }
          *(_DWORD *)(a1 + 8) = 0;
          v8 = 0;
          sub_CFC2E0(a1, v10, a3, a4, a5, (__int64)v7);
          v5 = *(_QWORD **)a2;
          v10 = *(unsigned int *)(a2 + 8);
          v9 = *(unsigned __int64 **)a1;
          v11 = *(_QWORD **)a2;
        }
        else
        {
          v11 = v5;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v8 *= 32LL;
            v24 = (unsigned __int64 *)((char *)v7 + v8);
            do
            {
              v25 = v7[2];
              v26 = v5[2];
              if ( v25 != v26 )
              {
                if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
                {
                  v38 = v7;
                  sub_BD60C0(v7);
                  v26 = v5[2];
                  v7 = v38;
                }
                v7[2] = v26;
                if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
                {
                  v39 = v7;
                  sub_BD6050(v7, *v5 & 0xFFFFFFFFFFFFFFF8LL);
                  v7 = v39;
                }
              }
              v27 = *((_DWORD *)v5 + 6);
              v7 += 4;
              v5 += 4;
              *((_DWORD *)v7 - 2) = v27;
            }
            while ( v7 != v24 );
            v5 = *(_QWORD **)a2;
            v10 = *(unsigned int *)(a2 + 8);
            v9 = *(unsigned __int64 **)a1;
            v11 = (_QWORD *)(*(_QWORD *)a2 + v8);
          }
        }
        v12 = (char *)v9 + v8;
        for ( i = &v5[4 * v10]; i != v11; v12 += 32 )
        {
          if ( v12 )
          {
            *(_QWORD *)v12 = 4;
            *((_QWORD *)v12 + 1) = 0;
            v14 = v11[2];
            *((_QWORD *)v12 + 2) = v14;
            if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
            {
              v35 = i;
              sub_BD6050((unsigned __int64 *)v12, *v11 & 0xFFFFFFFFFFFFFFF8LL);
              i = v35;
            }
            *((_DWORD *)v12 + 6) = *((_DWORD *)v11 + 6);
          }
          v11 += 4;
        }
        *(_DWORD *)(a1 + 8) = v33;
        v15 = *(_QWORD **)a2;
        v16 = (_QWORD *)(*(_QWORD *)a2 + 32LL * *(unsigned int *)(a2 + 8));
        if ( *(_QWORD **)a2 != v16 )
        {
          do
          {
            v17 = *(v16 - 2);
            v16 -= 4;
            if ( v17 != 0 && v17 != -4096 && v17 != -8192 )
              sub_BD60C0(v16);
          }
          while ( v15 != v16 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v18 = &v7[4 * v8];
      if ( v18 != v7 )
      {
        do
        {
          v19 = *(v18 - 2);
          v18 -= 4;
          if ( v19 != 0 && v19 != -4096 && v19 != -8192 )
          {
            v36 = v9;
            sub_BD60C0(v18);
            v9 = v36;
          }
        }
        while ( v18 != v9 );
        v7 = *(unsigned __int64 **)a1;
      }
      if ( v7 != (unsigned __int64 *)(a1 + 16) )
        _libc_free((unsigned __int64)v7);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v5;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
