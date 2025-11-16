// Function: sub_27589D0
// Address: 0x27589d0
//
void __fastcall sub_27589D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  unsigned __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  unsigned __int64 v12; // rcx
  int v13; // r8d
  unsigned __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // r14
  __int64 v17; // r12
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rbx
  unsigned __int64 v21; // r12
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  unsigned __int64 i; // rbx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  __int64 v27; // r12
  __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  __int64 v30; // rbx
  __int64 v31; // r15
  unsigned __int64 v32; // r12
  unsigned __int64 v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rcx
  unsigned __int64 j; // rbx
  unsigned __int64 v37; // rdi
  __int64 v38; // rbx
  __int64 v39; // r15
  unsigned __int64 v40; // r12
  unsigned __int64 v41; // rdi
  __int64 v42; // rax
  int v43; // [rsp-54h] [rbp-54h]
  unsigned __int64 v44; // [rsp-50h] [rbp-50h]
  __int64 v45; // [rsp-50h] [rbp-50h]
  unsigned __int64 v46; // [rsp-48h] [rbp-48h]
  __int64 v47; // [rsp-40h] [rbp-40h]
  __int64 v48; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v46 = *(unsigned int *)(a1 + 8);
    v47 = *(_QWORD *)a1;
    v44 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v9 = *(unsigned int *)(a2 + 8);
      v43 = v9;
      if ( v9 <= v46 )
      {
        v23 = *(_QWORD *)a1;
        if ( v9 )
        {
          v38 = a2 + 32;
          v39 = v47 + 16;
          do
          {
            v40 = *(_QWORD *)(v39 + 8);
            *(_QWORD *)(v39 - 16) = *(_QWORD *)(v38 - 16);
            while ( v40 )
            {
              sub_2754510(*(_QWORD *)(v40 + 24));
              v41 = v40;
              v40 = *(_QWORD *)(v40 + 16);
              j_j___libc_free_0(v41);
            }
            *(_QWORD *)(v39 + 8) = 0;
            *(_QWORD *)(v39 + 16) = v39;
            *(_QWORD *)(v39 + 24) = v39;
            *(_QWORD *)(v39 + 32) = 0;
            if ( *(_QWORD *)(v38 + 8) )
            {
              *(_DWORD *)v39 = *(_DWORD *)v38;
              v42 = *(_QWORD *)(v38 + 8);
              *(_QWORD *)(v39 + 8) = v42;
              *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
              *(_QWORD *)(v39 + 24) = *(_QWORD *)(v38 + 24);
              *(_QWORD *)(v42 + 8) = v39;
              *(_QWORD *)(v39 + 32) = *(_QWORD *)(v38 + 32);
              *(_QWORD *)(v38 + 8) = 0;
              *(_QWORD *)(v38 + 16) = v38;
              *(_QWORD *)(v38 + 24) = v38;
              *(_QWORD *)(v38 + 32) = 0;
            }
            v39 += 56;
            v38 += 56;
          }
          while ( v47 + 16 + 56 * v9 != v39 );
          v44 = 56 * v9 + v47;
          v23 = *(_QWORD *)a1;
          v46 = *(unsigned int *)(a1 + 8);
        }
        for ( i = v23 + 56 * v46; v44 != i; i -= 56LL )
        {
          v25 = *(_QWORD *)(i - 32);
          while ( v25 )
          {
            sub_2754510(*(_QWORD *)(v25 + 24));
            v26 = v25;
            v25 = *(_QWORD *)(v25 + 16);
            j_j___libc_free_0(v26);
          }
        }
        *(_DWORD *)(a1 + 8) = v9;
        v27 = *(_QWORD *)a2;
        v28 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v28 )
        {
          do
          {
            v29 = *(_QWORD *)(v28 - 32);
            v28 -= 56;
            sub_2754510(v29);
          }
          while ( v27 != v28 );
        }
        goto LABEL_17;
      }
      if ( v9 > *(unsigned int *)(a1 + 12) )
      {
        v35 = *(_QWORD *)a1;
        for ( j = v47 + 56 * v46; v47 != j; j -= 56LL )
        {
          v37 = *(_QWORD *)(j - 32);
          sub_2754510(v37);
        }
        *(_DWORD *)(a1 + 8) = 0;
        sub_2758850(a1, v9, a3, v35, a5, a6);
        v6 = *(_QWORD *)a2;
        v46 = 0;
        v9 = *(unsigned int *)(a2 + 8);
        v47 = *(_QWORD *)a1;
        v10 = *(_QWORD *)a2;
      }
      else
      {
        v10 = v6;
        if ( *(_DWORD *)(a1 + 8) )
        {
          v30 = a2 + 32;
          v31 = v47 + 16;
          v45 = 56 * v46;
          v46 = v45;
          v48 = v47 + 16 + v45;
          do
          {
            v32 = *(_QWORD *)(v31 + 8);
            *(_QWORD *)(v31 - 16) = *(_QWORD *)(v30 - 16);
            while ( v32 )
            {
              sub_2754510(*(_QWORD *)(v32 + 24));
              v33 = v32;
              v32 = *(_QWORD *)(v32 + 16);
              j_j___libc_free_0(v33);
            }
            *(_QWORD *)(v31 + 8) = 0;
            *(_QWORD *)(v31 + 16) = v31;
            *(_QWORD *)(v31 + 24) = v31;
            *(_QWORD *)(v31 + 32) = 0;
            if ( *(_QWORD *)(v30 + 8) )
            {
              *(_DWORD *)v31 = *(_DWORD *)v30;
              v34 = *(_QWORD *)(v30 + 8);
              *(_QWORD *)(v31 + 8) = v34;
              *(_QWORD *)(v31 + 16) = *(_QWORD *)(v30 + 16);
              *(_QWORD *)(v31 + 24) = *(_QWORD *)(v30 + 24);
              *(_QWORD *)(v34 + 8) = v31;
              *(_QWORD *)(v31 + 32) = *(_QWORD *)(v30 + 32);
              *(_QWORD *)(v30 + 8) = 0;
              *(_QWORD *)(v30 + 16) = v30;
              *(_QWORD *)(v30 + 24) = v30;
              *(_QWORD *)(v30 + 32) = 0;
            }
            v31 += 56;
            v30 += 56;
          }
          while ( v31 != v48 );
          v6 = *(_QWORD *)a2;
          v9 = *(unsigned int *)(a2 + 8);
          v10 = *(_QWORD *)a2 + v45;
          v47 = *(_QWORD *)a1;
        }
      }
      v11 = v46 + v47;
      v12 = v6 + 56 * v9;
      if ( v12 == v10 )
      {
LABEL_13:
        *(_DWORD *)(a1 + 8) = v43;
        v16 = *(_QWORD *)a2;
        v17 = *(_QWORD *)a2 + 56LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v17 )
        {
          do
          {
            v18 = *(_QWORD *)(v17 - 32);
            v17 -= 56;
            while ( v18 )
            {
              sub_2754510(*(_QWORD *)(v18 + 24));
              v19 = v18;
              v18 = *(_QWORD *)(v18 + 16);
              j_j___libc_free_0(v19);
            }
          }
          while ( v16 != v17 );
        }
LABEL_17:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      while ( 1 )
      {
        if ( !v11 )
          goto LABEL_9;
        v14 = v11 + 16;
        *(_QWORD *)v11 = *(_QWORD *)v10;
        v15 = *(_QWORD *)(v10 + 24);
        if ( v15 )
        {
          v13 = *(_DWORD *)(v10 + 16);
          *(_QWORD *)(v11 + 24) = v15;
          *(_DWORD *)(v11 + 16) = v13;
          *(_QWORD *)(v11 + 32) = *(_QWORD *)(v10 + 32);
          *(_QWORD *)(v11 + 40) = *(_QWORD *)(v10 + 40);
          *(_QWORD *)(v15 + 8) = v14;
          *(_QWORD *)(v11 + 48) = *(_QWORD *)(v10 + 48);
          *(_QWORD *)(v10 + 24) = 0;
          *(_QWORD *)(v10 + 32) = v10 + 16;
          *(_QWORD *)(v10 + 40) = v10 + 16;
          *(_QWORD *)(v10 + 48) = 0;
LABEL_9:
          v10 += 56;
          v11 += 56LL;
          if ( v12 == v10 )
            goto LABEL_13;
        }
        else
        {
          v10 += 56;
          *(_DWORD *)(v11 + 16) = 0;
          v11 += 56LL;
          *(_QWORD *)(v11 - 32) = 0;
          *(_QWORD *)(v11 - 24) = v14;
          *(_QWORD *)(v11 - 16) = v14;
          *(_QWORD *)(v11 - 8) = 0;
          if ( v12 == v10 )
            goto LABEL_13;
        }
      }
    }
    v20 = v47 + 56 * v46;
    if ( v47 != v20 )
    {
      do
      {
        v21 = *(_QWORD *)(v20 - 32);
        v20 -= 56LL;
        while ( v21 )
        {
          sub_2754510(*(_QWORD *)(v21 + 24));
          v22 = v21;
          v21 = *(_QWORD *)(v21 + 16);
          j_j___libc_free_0(v22);
        }
      }
      while ( v47 != v20 );
      v20 = *(_QWORD *)a1;
    }
    if ( v20 != a1 + 16 )
      _libc_free(v20);
    *(_QWORD *)a1 = *(_QWORD *)a2;
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
    *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
    *(_QWORD *)a2 = v6;
    *(_QWORD *)(a2 + 8) = 0;
  }
}
