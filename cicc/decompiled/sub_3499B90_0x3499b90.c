// Function: sub_3499B90
// Address: 0x3499b90
//
void __fastcall sub_3499B90(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // rbx
  __int64 v7; // r14
  int *i; // r12
  int v9; // eax
  __int64 v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // r14
  __int64 v14; // rbx
  unsigned __int64 v15; // r15
  unsigned __int64 *v16; // rbx
  __int64 v17; // r12
  __int64 v18; // rax
  unsigned __int64 v19; // r14
  unsigned __int64 *v20; // r15
  __int64 v21; // r15
  __int64 v22; // r12
  __int64 v23; // r14
  int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rbx
  __int64 v28; // r15
  unsigned __int64 v29; // r12
  int v30; // edx
  __int64 v31; // rsi
  __int64 v32; // rdi
  unsigned int v33; // [rsp+Ch] [rbp-44h]
  __int64 v35; // [rsp+18h] [rbp-38h]
  __int64 v36; // [rsp+18h] [rbp-38h]
  unsigned __int64 v37; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    v2 = *(_QWORD *)a1;
    v33 = *(_DWORD *)(a2 + 8);
    v4 = *(_QWORD *)a1;
    v35 = v33;
    v5 = *(unsigned int *)(a1 + 8);
    if ( v33 > v5 )
    {
      if ( v33 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v17 = v2 + 56 * v5;
        while ( v2 != v17 )
        {
          while ( 1 )
          {
            v18 = *(unsigned int *)(v17 - 40);
            v19 = *(_QWORD *)(v17 - 48);
            v17 -= 56;
            v18 *= 32;
            v20 = (unsigned __int64 *)(v19 + v18);
            if ( v19 != v19 + v18 )
            {
              do
              {
                v20 -= 4;
                if ( (unsigned __int64 *)*v20 != v20 + 2 )
                  j_j___libc_free_0(*v20);
              }
              while ( (unsigned __int64 *)v19 != v20 );
              v19 = *(_QWORD *)(v17 + 8);
            }
            if ( v19 == v17 + 24 )
              break;
            _libc_free(v19);
            if ( v2 == v17 )
              goto LABEL_33;
          }
        }
LABEL_33:
        *(_DWORD *)(a1 + 8) = 0;
        sub_B3C890(a1, v33);
        v2 = *(_QWORD *)a1;
        v35 = *(unsigned int *)(a2 + 8);
        v5 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v27 = v2 + 8;
        v5 *= 56LL;
        v28 = *(_QWORD *)a2 + 8LL;
        v29 = v28 + v5;
        do
        {
          v30 = *(_DWORD *)(v28 - 8);
          v31 = v28;
          v32 = v27;
          v28 += 56;
          v37 = v5;
          v27 += 56;
          *(_DWORD *)(v27 - 64) = v30;
          sub_34994E0(v32, v31);
          v5 = v37;
        }
        while ( v29 != v28 );
        v2 = *(_QWORD *)a1;
        v35 = *(unsigned int *)(a2 + 8);
      }
      v6 = v5 + v2;
      v7 = *(_QWORD *)a2 + 56 * v35;
      for ( i = (int *)(v5 + *(_QWORD *)a2); (int *)v7 != i; i += 14 )
      {
        while ( 1 )
        {
          if ( v6 )
          {
            v9 = *i;
            *(_DWORD *)(v6 + 16) = 0;
            *(_DWORD *)(v6 + 20) = 1;
            *(_DWORD *)v6 = v9;
            *(_QWORD *)(v6 + 8) = v6 + 24;
            if ( i[4] )
              break;
          }
          i += 14;
          v6 += 56;
          if ( (int *)v7 == i )
            goto LABEL_11;
        }
        v10 = (__int64)(i + 2);
        v11 = v6 + 8;
        v6 += 56;
        sub_34994E0(v11, v10);
      }
      goto LABEL_11;
    }
    v12 = *(_QWORD *)a1;
    if ( v33 )
    {
      v21 = *(_QWORD *)a2 + 8LL;
      v22 = v2 + 8;
      v36 = 56LL * v33;
      v23 = v21 + v36;
      do
      {
        v24 = *(_DWORD *)(v21 - 8);
        v25 = v21;
        v26 = v22;
        v21 += 56;
        v22 += 56;
        *(_DWORD *)(v22 - 64) = v24;
        sub_34994E0(v26, v25);
      }
      while ( v23 != v21 );
      v12 = *(_QWORD *)a1;
      v5 = *(unsigned int *)(a1 + 8);
      v4 = v2 + v36;
    }
    v13 = v12 + 56 * v5;
    if ( v13 == v4 )
    {
LABEL_11:
      *(_DWORD *)(a1 + 8) = v33;
      return;
    }
    do
    {
      v14 = *(unsigned int *)(v13 - 40);
      v15 = *(_QWORD *)(v13 - 48);
      v13 -= 56;
      v16 = (unsigned __int64 *)(v15 + 32 * v14);
      if ( (unsigned __int64 *)v15 != v16 )
      {
        do
        {
          v16 -= 4;
          if ( (unsigned __int64 *)*v16 != v16 + 2 )
            j_j___libc_free_0(*v16);
        }
        while ( (unsigned __int64 *)v15 != v16 );
        v15 = *(_QWORD *)(v13 + 8);
      }
      if ( v15 != v13 + 24 )
        _libc_free(v15);
    }
    while ( v4 != v13 );
    *(_DWORD *)(a1 + 8) = v33;
  }
}
