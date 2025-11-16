// Function: sub_2706C70
// Address: 0x2706c70
//
void __fastcall sub_2706C70(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v8; // rsi
  __int64 v9; // rbx
  unsigned __int64 v10; // rdx
  int v11; // r14d
  __int64 v12; // r12
  __int64 v13; // r12
  __int64 v14; // rbx
  __int64 v15; // r15
  __int64 *i; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rbx
  unsigned __int64 v20; // rdi
  __int64 v21; // r12
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // r12
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 v29; // rdi
  __int64 v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rdi
  __int64 v35; // [rsp-48h] [rbp-48h]
  unsigned __int64 v36; // [rsp-48h] [rbp-48h]
  __int64 v37; // [rsp-40h] [rbp-40h]
  unsigned __int64 v38; // [rsp-40h] [rbp-40h]

  if ( (__int64 *)a1 != a2 )
  {
    v8 = *((unsigned int *)a2 + 2);
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v11 = v8;
    v12 = *(_QWORD *)a1;
    if ( v8 <= v10 )
    {
      v18 = *(_QWORD *)a1;
      if ( v8 )
      {
        v23 = *a2;
        v24 = v9 + 8;
        v25 = 80 * v8;
        v26 = v23 + 8;
        v35 = 80 * v8;
        v37 = v23 + 8 + 80 * v8;
        do
        {
          v27 = *(_QWORD *)(v26 - 8);
          v28 = v26;
          v29 = v24;
          v26 += 80;
          v24 += 80;
          *(_QWORD *)(v24 - 88) = v27;
          sub_26F6330(v29, v28, v25, v27, a5, a6);
          *(_DWORD *)(v24 - 16) = *(_DWORD *)(v26 - 16);
        }
        while ( v37 != v26 );
        v18 = *(_QWORD *)a1;
        v10 = *(unsigned int *)(a1 + 8);
        v12 = v9 + v35;
      }
      v19 = v18 + 80 * v10;
      while ( v12 != v19 )
      {
        v19 -= 80;
        v20 = *(_QWORD *)(v19 + 8);
        if ( v20 != v19 + 24 )
          _libc_free(v20);
      }
    }
    else
    {
      if ( v8 > *(unsigned int *)(a1 + 12) )
      {
        v21 = v9 + 80 * v10;
        while ( v21 != v9 )
        {
          while ( 1 )
          {
            v21 -= 80;
            v22 = *(_QWORD *)(v21 + 8);
            if ( v22 == v21 + 24 )
              break;
            _libc_free(v22);
            if ( v21 == v9 )
              goto LABEL_22;
          }
        }
LABEL_22:
        *(_DWORD *)(a1 + 8) = 0;
        sub_F30F50(a1, v8, v10, a4, a5, a6);
        v8 = *((unsigned int *)a2 + 2);
        v9 = *(_QWORD *)a1;
        v10 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v30 = v9 + 8;
        v10 *= 80LL;
        v31 = *a2 + 8;
        v36 = v31 + v10;
        do
        {
          v32 = *(_QWORD *)(v31 - 8);
          v33 = v31;
          v34 = v30;
          v38 = v10;
          v30 += 80;
          v31 += 80;
          *(_QWORD *)(v30 - 88) = v32;
          sub_26F6330(v34, v33, v10, v32, a5, a6);
          a4 = *(unsigned int *)(v31 - 16);
          v10 = v38;
          *(_DWORD *)(v30 - 16) = a4;
        }
        while ( v31 != v36 );
        v8 = *((unsigned int *)a2 + 2);
        v9 = *(_QWORD *)a1;
      }
      v13 = *a2;
      v14 = v10 + v9;
      v15 = *a2 + 80 * v8;
      for ( i = (__int64 *)(v10 + v13); (__int64 *)v15 != i; v14 += 80 )
      {
        if ( v14 )
        {
          v17 = *i;
          *(_DWORD *)(v14 + 16) = 0;
          *(_DWORD *)(v14 + 20) = 6;
          *(_QWORD *)v14 = v17;
          *(_QWORD *)(v14 + 8) = v14 + 24;
          if ( *((_DWORD *)i + 4) )
            sub_26F6330(v14 + 8, (__int64)(i + 1), v10, a4, a5, a6);
          *(_DWORD *)(v14 + 72) = *((_DWORD *)i + 18);
        }
        i += 10;
      }
    }
    *(_DWORD *)(a1 + 8) = v11;
  }
}
