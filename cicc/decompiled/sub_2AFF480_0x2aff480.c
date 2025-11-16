// Function: sub_2AFF480
// Address: 0x2aff480
//
void __fastcall sub_2AFF480(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  int v11; // r14d
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 *v15; // r15
  __int64 *i; // r12
  __int64 v17; // rax
  __int64 v18; // rsi
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rbx
  unsigned __int64 v22; // rdi
  __int64 v23; // r15
  unsigned __int64 v24; // rdi
  __int64 v25; // rdx
  __int64 v26; // r15
  __int64 v27; // r12
  __int64 v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rdi
  __int64 v31; // rbx
  __int64 v32; // r15
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // [rsp-48h] [rbp-48h]
  unsigned __int64 v37; // [rsp-48h] [rbp-48h]
  unsigned __int64 v38; // [rsp-40h] [rbp-40h]
  unsigned __int64 v39; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *((unsigned int *)a2 + 2);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = v8;
    v12 = *(_QWORD *)a1;
    if ( v8 <= v10 )
    {
      v20 = *(_QWORD *)a1;
      if ( v8 )
      {
        v25 = 5 * v8;
        v26 = v9 + 8;
        v27 = *a2 + 8;
        v36 = 88 * v8;
        v38 = v27 + 88 * v8;
        do
        {
          v28 = *(_QWORD *)(v27 - 8);
          v29 = v27;
          v30 = v26;
          v27 += 88;
          v26 += 88;
          *(_QWORD *)(v26 - 96) = v28;
          sub_2AF6C10(v30, v29, v25, v28, a5, a6);
        }
        while ( v38 != v27 );
        v20 = *(_QWORD *)a1;
        v10 = *(unsigned int *)(a1 + 8);
        v12 = v9 + v36;
      }
      v21 = v20 + 88 * v10;
      while ( v12 != v21 )
      {
        v21 -= 88;
        v22 = *(_QWORD *)(v21 + 8);
        if ( v22 != v21 + 24 )
          _libc_free(v22);
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        v23 = v9 + 88 * v10;
        while ( v23 != v9 )
        {
          while ( 1 )
          {
            v23 -= 88;
            v24 = *(_QWORD *)(v23 + 8);
            if ( v24 == v23 + 24 )
              break;
            _libc_free(v24);
            if ( v23 == v9 )
              goto LABEL_22;
          }
        }
LABEL_22:
        *(_DWORD *)(a1 + 8) = 0;
        sub_2AFF360(a1, v8, v10, a4, a5, a6);
        v8 = *((unsigned int *)a2 + 2);
        v9 = *(_QWORD *)a1;
        v10 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v31 = v9 + 8;
        v10 *= 88LL;
        v32 = *a2 + 8;
        v37 = v32 + v10;
        do
        {
          v33 = *(_QWORD *)(v32 - 8);
          v34 = v32;
          v35 = v31;
          v39 = v10;
          v31 += 88;
          v32 += 88;
          *(_QWORD *)(v31 - 96) = v33;
          sub_2AF6C10(v35, v34, v10, v33, a5, a6);
          v10 = v39;
        }
        while ( v37 != v32 );
        v8 = *((unsigned int *)a2 + 2);
        v9 = *(_QWORD *)a1;
      }
      v13 = *a2;
      v14 = v10 + v9;
      v15 = (__int64 *)(v13 + 88 * v8);
      for ( i = (__int64 *)(v10 + v13); v15 != i; i += 11 )
      {
        while ( 1 )
        {
          if ( v14 )
          {
            v17 = *i;
            *(_DWORD *)(v14 + 16) = 0;
            *(_DWORD *)(v14 + 20) = 8;
            *(_QWORD *)v14 = v17;
            *(_QWORD *)(v14 + 8) = v14 + 24;
            if ( *((_DWORD *)i + 4) )
              break;
          }
          i += 11;
          v14 += 88;
          if ( v15 == i )
            goto LABEL_11;
        }
        v18 = (__int64)(i + 1);
        v19 = v14 + 8;
        v14 += 88;
        sub_2AF6C10(v19, v18, v10, a4, a5, a6);
      }
    }
LABEL_11:
    *(_DWORD *)(a1 + 8) = v11;
  }
}
