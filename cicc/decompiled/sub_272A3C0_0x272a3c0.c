// Function: sub_272A3C0
// Address: 0x272a3c0
//
void __fastcall sub_272A3C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r12
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // r8
  unsigned __int64 v12; // rsi
  int v13; // r15d
  __int64 v14; // rax
  unsigned __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // r12
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rbx
  unsigned __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v29; // rbx
  unsigned __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r12
  __int64 v33; // rdi
  __int64 v34; // rsi
  unsigned __int64 v35; // r12
  unsigned __int64 v36; // rdi
  __int64 v37; // rdx
  __int64 v38; // r12
  __int64 v39; // rdi
  __int64 v40; // [rsp-50h] [rbp-50h]
  unsigned __int64 v41; // [rsp-48h] [rbp-48h]
  __int64 v42; // [rsp-40h] [rbp-40h]
  unsigned __int64 v43; // [rsp-40h] [rbp-40h]
  unsigned __int64 v44; // [rsp-40h] [rbp-40h]
  __int64 v45; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = a2 + 16;
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v12 = *(unsigned int *)(a2 + 8);
      v13 = v12;
      if ( v12 <= v10 )
      {
        v25 = *(_QWORD *)a1;
        if ( v12 )
        {
          v37 = a2 + 24;
          v38 = v9 + 8;
          do
          {
            v39 = v38;
            v45 = v37;
            v38 += 56;
            *(_QWORD *)(v38 - 64) = *(_QWORD *)(v37 - 8);
            sub_2729AE0(v39, v37, v37, a4, v11, a6);
            v37 = v45 + 56;
          }
          while ( v9 + 8 + 56 * v12 != v38 );
          v25 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a1 + 8);
          v11 = v9 + 56 * v12;
        }
        v26 = v25 + 56 * v10;
        while ( v11 != v26 )
        {
          v26 -= 56LL;
          v27 = *(_QWORD *)(v26 + 8);
          if ( v27 != v26 + 24 )
          {
            v43 = v11;
            _libc_free(v27);
            v11 = v43;
          }
        }
        *(_DWORD *)(a1 + 8) = v12;
        v28 = *(_QWORD *)a2;
        v29 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v29 )
        {
          do
          {
            v29 -= 56;
            v30 = *(_QWORD *)(v29 + 8);
            if ( v30 != v29 + 24 )
              _libc_free(v30);
          }
          while ( v28 != v29 );
        }
      }
      else
      {
        if ( v12 > *(unsigned int *)(a1 + 12) )
        {
          v35 = v9 + 56 * v10;
          while ( v35 != v9 )
          {
            while ( 1 )
            {
              v35 -= 56LL;
              v36 = *(_QWORD *)(v35 + 8);
              if ( v36 == v35 + 24 )
                break;
              _libc_free(v36);
              if ( v35 == v9 )
                goto LABEL_43;
            }
          }
LABEL_43:
          *(_DWORD *)(a1 + 8) = 0;
          sub_272A2A0(a1, v12, v10, a4, v11, a6);
          v8 = *(_QWORD *)a2;
          v12 = *(unsigned int *)(a2 + 8);
          v10 = 0;
          v9 = *(_QWORD *)a1;
          v14 = *(_QWORD *)a2;
        }
        else
        {
          v14 = v8;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v31 = v9 + 8;
            v32 = a2 + 24;
            v40 = 56 * v10;
            v10 *= 56LL;
            v41 = v31 + v10;
            do
            {
              v33 = v31;
              v44 = v10;
              v31 += 56;
              *(_QWORD *)(v31 - 64) = *(_QWORD *)(v32 - 8);
              v34 = v32;
              v32 += 56;
              sub_2729AE0(v33, v34, v10, a4, v11, a6);
              v10 = v44;
            }
            while ( v31 != v41 );
            v8 = *(_QWORD *)a2;
            v12 = *(unsigned int *)(a2 + 8);
            v9 = *(_QWORD *)a1;
            v14 = *(_QWORD *)a2 + v40;
          }
        }
        v15 = v10 + v9;
        v16 = v8 + 56 * v12;
        v17 = v14;
        if ( v16 != v14 )
        {
          do
          {
            while ( 1 )
            {
              if ( v15 )
              {
                v18 = *(_QWORD *)v17;
                *(_DWORD *)(v15 + 16) = 0;
                *(_DWORD *)(v15 + 20) = 2;
                *(_QWORD *)v15 = v18;
                *(_QWORD *)(v15 + 8) = v15 + 24;
                if ( *(_DWORD *)(v17 + 16) )
                  break;
              }
              v17 += 56;
              v15 += 56LL;
              if ( v16 == v17 )
                goto LABEL_12;
            }
            v19 = v17 + 8;
            v42 = v16;
            v17 += 56;
            sub_2729AE0(v15 + 8, v19, v16, a4, v11, a6);
            v16 = v42;
            v15 += 56LL;
          }
          while ( v42 != v17 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v13;
        v20 = *(_QWORD *)a2;
        v21 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v21 )
        {
          do
          {
            v21 -= 56;
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 != v21 + 24 )
              _libc_free(v22);
          }
          while ( v20 != v21 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v23 = v9 + 56 * v10;
      if ( v23 != v9 )
      {
        do
        {
          v23 -= 56LL;
          v24 = *(_QWORD *)(v23 + 8);
          if ( v24 != v23 + 24 )
            _libc_free(v24);
        }
        while ( v23 != v9 );
        v11 = *(_QWORD *)a1;
      }
      if ( v11 != a1 + 16 )
        _libc_free(v11);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)a2 = v8;
      *(_QWORD *)(a2 + 8) = 0;
    }
  }
}
