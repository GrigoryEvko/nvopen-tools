// Function: sub_31FDD40
// Address: 0x31fdd40
//
void __fastcall sub_31FDD40(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // r14
  unsigned __int64 v9; // rbx
  unsigned __int64 v10; // r8
  unsigned __int64 v11; // rsi
  int v12; // r15d
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rcx
  unsigned __int64 v16; // rbx
  __int64 v17; // r14
  unsigned __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rsi
  __int64 v22; // r13
  __int64 v23; // rbx
  unsigned __int64 v24; // rdi
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rdx
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 v30; // r13
  __int64 v31; // rbx
  unsigned __int64 v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // r14
  __int64 v35; // rsi
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // rsi
  __int64 v41; // [rsp-50h] [rbp-50h]
  __int64 v42; // [rsp-50h] [rbp-50h]
  __int64 v43; // [rsp-48h] [rbp-48h]
  __int64 v44; // [rsp-40h] [rbp-40h]
  unsigned __int64 v45; // [rsp-40h] [rbp-40h]
  unsigned __int64 v46; // [rsp-40h] [rbp-40h]
  __int64 v47; // [rsp-40h] [rbp-40h]
  __int64 v48; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v8 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v44 = a2 + 16;
    v10 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v12 = v11;
      if ( v11 <= v9 )
      {
        v27 = *(_QWORD *)a1;
        if ( v11 )
        {
          v38 = a2 + 24;
          v39 = v8 + 8;
          v42 = 40 * v11;
          v43 = a2 + 24 + 40 * v11;
          do
          {
            v48 = v39;
            *(_QWORD *)(v39 - 8) = *(_QWORD *)(v38 - 8);
            v40 = v38;
            v38 += 40;
            sub_31F4130(v39, v40, v39, a4, v10, a6);
            v39 = v48 + 40;
          }
          while ( v43 != v38 );
          v27 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a1 + 8);
          v10 = v8 + v42;
        }
        v28 = v27 + 40 * v9;
        while ( v10 != v28 )
        {
          v28 -= 40LL;
          v29 = *(_QWORD *)(v28 + 8);
          if ( v29 != v28 + 24 )
          {
            v46 = v10;
            _libc_free(v29);
            v10 = v46;
          }
        }
        *(_DWORD *)(a1 + 8) = v12;
        v30 = *(_QWORD *)a2;
        v31 = *(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v31 )
        {
          do
          {
            v31 -= 40;
            v32 = *(_QWORD *)(v31 + 8);
            if ( v32 != v31 + 24 )
              _libc_free(v32);
          }
          while ( v30 != v31 );
        }
      }
      else
      {
        v13 = *(unsigned int *)(a1 + 12);
        if ( v11 > v13 )
        {
          v36 = v8 + 40 * v9;
          while ( v36 != v8 )
          {
            while ( 1 )
            {
              v36 -= 40LL;
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
          sub_31FDC20(a1, v11, v13, a4, v10, a6);
          v11 = *(unsigned int *)(a2 + 8);
          v8 = *(_QWORD *)a1;
          v44 = *(_QWORD *)a2;
          v14 = *(_QWORD *)a2;
        }
        else
        {
          v14 = v44;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v33 = v8 + 8;
            v34 = a2 + 24;
            v41 = 40 * v9;
            v9 *= 40LL;
            do
            {
              v47 = v33;
              *(_QWORD *)(v33 - 8) = *(_QWORD *)(v34 - 8);
              v35 = v34;
              v34 += 40;
              sub_31F4130(v33, v35, v14, v33, v10, a6);
              v33 = v47 + 40;
            }
            while ( v34 != a2 + 24 + v9 );
            v11 = *(unsigned int *)(a2 + 8);
            v8 = *(_QWORD *)a1;
            v44 = *(_QWORD *)a2;
            v14 = *(_QWORD *)a2 + v41;
          }
        }
        v15 = v44;
        v16 = v8 + v9;
        v17 = v14;
        v18 = v44 + 40 * v11;
        if ( v18 != v14 )
        {
          do
          {
            while ( 1 )
            {
              if ( v16 )
              {
                v19 = *(_QWORD *)v17;
                *(_DWORD *)(v16 + 16) = 0;
                *(_DWORD *)(v16 + 20) = 1;
                *(_QWORD *)v16 = v19;
                *(_QWORD *)(v16 + 8) = v16 + 24;
                v20 = *(unsigned int *)(v17 + 16);
                if ( (_DWORD)v20 )
                  break;
              }
              v17 += 40;
              v16 += 40LL;
              if ( v18 == v17 )
                goto LABEL_12;
            }
            v21 = v17 + 8;
            v45 = v18;
            v17 += 40;
            sub_31F4130(v16 + 8, v21, v20, v15, v10, a6);
            v18 = v45;
            v16 += 40LL;
          }
          while ( v45 != v17 );
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = v12;
        v22 = *(_QWORD *)a2;
        v23 = *(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v23 )
        {
          do
          {
            v23 -= 40;
            v24 = *(_QWORD *)(v23 + 8);
            if ( v24 != v23 + 24 )
              _libc_free(v24);
          }
          while ( v22 != v23 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v25 = v8 + 40 * v9;
      if ( v25 != v8 )
      {
        do
        {
          v25 -= 40LL;
          v26 = *(_QWORD *)(v25 + 8);
          if ( v26 != v25 + 24 )
            _libc_free(v26);
        }
        while ( v25 != v8 );
        v10 = *(_QWORD *)a1;
      }
      if ( v10 != a1 + 16 )
        _libc_free(v10);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v44;
    }
  }
}
