// Function: sub_2707240
// Address: 0x2707240
//
void __fastcall sub_2707240(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 i; // r12
  __int64 v16; // r14
  __int64 v17; // r8
  __int64 v18; // rdx
  unsigned __int64 v19; // r14
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // r15
  __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  __int64 v24; // rcx
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rbx
  __int64 v28; // r14
  unsigned __int64 v29; // rdi
  __int64 *v30; // rbx
  __int64 v31; // r14
  __int64 v32; // rdi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rcx
  __int64 *v38; // r15
  unsigned __int64 v39; // rbx
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  unsigned int v45; // [rsp+Ch] [rbp-44h]
  __int64 v46; // [rsp+18h] [rbp-38h]
  unsigned __int64 v47; // [rsp+18h] [rbp-38h]

  if ( a1 != a2 )
  {
    v7 = *(_QWORD *)a1;
    v9 = *(unsigned int *)(a1 + 8);
    v45 = *(_DWORD *)(a2 + 8);
    v6 = v45;
    v46 = *(_QWORD *)a1;
    if ( v45 <= v9 )
    {
      v18 = *(_QWORD *)a1;
      if ( v45 )
      {
        v30 = *(__int64 **)a2;
        v31 = v7 + 40LL * v45;
        do
        {
          sub_2706C70(v7, v30, v18, a4, a5, a6);
          v32 = v7 + 16;
          v7 += 40;
          sub_26F6260(v32, (__int64)(v30 + 2), v33, v34, v35, v36);
          v30 += 5;
        }
        while ( v31 != v7 );
        v18 = *(_QWORD *)a1;
        v9 = *(unsigned int *)(a1 + 8);
      }
      v19 = v18 + 40 * v9;
      while ( v7 != v19 )
      {
        v19 -= 40LL;
        v20 = *(_QWORD *)(v19 + 16);
        if ( v20 != v19 + 40 )
          _libc_free(v20);
        v21 = *(_QWORD *)v19;
        v22 = *(_QWORD *)v19 + 80LL * *(unsigned int *)(v19 + 8);
        if ( *(_QWORD *)v19 != v22 )
        {
          do
          {
            v22 -= 80;
            v23 = *(_QWORD *)(v22 + 8);
            if ( v23 != v22 + 24 )
              _libc_free(v23);
          }
          while ( v21 != v22 );
          v21 = *(_QWORD *)v19;
        }
        if ( v21 != v19 + 16 )
          _libc_free(v21);
      }
    }
    else
    {
      v10 = *(unsigned int *)(a1 + 12);
      if ( v45 > v10 )
      {
        v24 = *(_QWORD *)a1;
        v25 = v46 + 40 * v9;
        while ( v25 != v46 )
        {
          v25 -= 40LL;
          v26 = *(_QWORD *)(v25 + 16);
          if ( v26 != v25 + 40 )
            _libc_free(v26);
          v27 = *(_QWORD *)v25;
          v28 = *(_QWORD *)v25 + 80LL * *(unsigned int *)(v25 + 8);
          if ( *(_QWORD *)v25 != v28 )
          {
            do
            {
              v28 -= 80;
              v29 = *(_QWORD *)(v28 + 8);
              v10 = v28 + 24;
              if ( v29 != v28 + 24 )
                _libc_free(v29);
            }
            while ( v27 != v28 );
            v27 = *(_QWORD *)v25;
          }
          if ( v27 != v25 + 16 )
            _libc_free(v27);
        }
        *(_DWORD *)(a1 + 8) = 0;
        sub_F31630(a1, v45, v10, v24, a5, a6);
        v6 = *(unsigned int *)(a2 + 8);
        v46 = *(_QWORD *)a1;
        v9 = 0;
      }
      else if ( *(_DWORD *)(a1 + 8) )
      {
        v37 = a2;
        v9 *= 40LL;
        v38 = *(__int64 **)a2;
        v39 = v7 + v9;
        do
        {
          v47 = v9;
          sub_2706C70(v7, v38, v10, v37, a5, a6);
          v40 = v7 + 16;
          v7 += 40;
          sub_26F6260(v40, (__int64)(v38 + 2), v41, v42, v43, v44);
          v38 += 5;
          v9 = v47;
        }
        while ( v7 != v39 );
        v6 = *(unsigned int *)(a2 + 8);
        v46 = *(_QWORD *)a1;
      }
      v11 = a2;
      v12 = 5 * v6;
      v13 = v9 + v46;
      v14 = *(_QWORD *)a2 + 40 * v6;
      for ( i = v9 + *(_QWORD *)a2; v14 != i; v13 = v16 )
      {
        v16 = 40;
        if ( v13 )
        {
          v17 = v13 + 16;
          *(_DWORD *)(v13 + 8) = 0;
          *(_QWORD *)v13 = v13 + 16;
          *(_DWORD *)(v13 + 12) = 0;
          if ( *(_DWORD *)(i + 8) )
          {
            sub_2706C70(v13, (__int64 *)i, v12, v11, v17, a6);
            v17 = v13 + 16;
          }
          v16 = v13 + 40;
          *(_QWORD *)(v13 + 24) = 0;
          *(_QWORD *)(v13 + 16) = v13 + 40;
          *(_QWORD *)(v13 + 32) = 0;
          if ( *(_QWORD *)(i + 24) )
            sub_26F6260(v17, i + 16, v12, v11, v17, a6);
        }
        i += 40LL;
      }
    }
    *(_DWORD *)(a1 + 8) = v45;
  }
}
