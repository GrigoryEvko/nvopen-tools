// Function: sub_2732900
// Address: 0x2732900
//
void __fastcall sub_2732900(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rcx
  unsigned __int64 v9; // rsi
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  __int64 v13; // r15
  __int64 v14; // rcx
  unsigned __int64 v15; // r12
  unsigned __int64 i; // rbx
  __int64 v17; // rax
  __int64 v18; // r14
  __int64 v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // r15
  unsigned __int64 *v22; // rbx
  unsigned __int64 v23; // rbx
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  unsigned __int64 *v26; // r15
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rax
  unsigned __int64 v30; // r15
  unsigned __int64 *v31; // r12
  __int64 v32; // r14
  __int64 v33; // r12
  __int64 v34; // rax
  unsigned __int64 v35; // r15
  unsigned __int64 *v36; // rbx
  __int64 v37; // r12
  __int64 v38; // r15
  unsigned __int64 v39; // rbx
  __int64 v40; // rcx
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rcx
  __int64 v44; // rdx
  unsigned __int64 v45; // r15
  __int64 v46; // rax
  unsigned __int64 v47; // rbx
  unsigned __int64 *v48; // r12
  __int64 v49; // rbx
  __int64 v50; // r12
  __int64 v51; // r15
  __int64 v52; // rcx
  __int64 v53; // rsi
  __int64 v54; // rdi
  __int64 v55; // rcx
  int v56; // [rsp-5Ch] [rbp-5Ch]
  unsigned __int64 v57; // [rsp-58h] [rbp-58h]
  __int64 v58; // [rsp-50h] [rbp-50h]
  unsigned __int64 v59; // [rsp-48h] [rbp-48h]
  __int64 v60; // [rsp-48h] [rbp-48h]
  __int64 v61; // [rsp-48h] [rbp-48h]
  unsigned __int64 v62; // [rsp-40h] [rbp-40h]
  unsigned __int64 v63; // [rsp-40h] [rbp-40h]

  if ( a1 != a2 )
  {
    v6 = a2 + 16;
    v9 = *(_QWORD *)a1;
    v10 = *(unsigned int *)(a1 + 8);
    v58 = v6;
    v62 = *(_QWORD *)a1;
    v59 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == v6 )
    {
      v11 = *(unsigned int *)(a2 + 8);
      v56 = *(_DWORD *)(a2 + 8);
      v57 = v11;
      if ( v11 <= v10 )
      {
        v27 = *(_QWORD *)a1;
        if ( *(_DWORD *)(a2 + 8) )
        {
          v49 = a2 + 32;
          v50 = 672 * v11;
          v51 = v62 + 16;
          v61 = v62 + 16 + 672 * v11;
          do
          {
            v52 = *(_QWORD *)(v49 - 16);
            v53 = v49;
            v54 = v51;
            v49 += 672;
            v51 += 672;
            *(_QWORD *)(v51 - 688) = v52;
            v55 = *(_QWORD *)(v49 - 680);
            *(_QWORD *)(v51 - 680) = v55;
            sub_2731860(v54, v53, v27, v55, a5, a6);
          }
          while ( v61 != v51 );
          v27 = *(_QWORD *)a1;
          v59 = v62 + v50;
          v10 = *(unsigned int *)(a1 + 8);
        }
        v28 = v27 + 672 * v10;
        while ( v59 != v28 )
        {
          v29 = *(unsigned int *)(v28 - 648);
          v30 = *(_QWORD *)(v28 - 656);
          v28 -= 672;
          v31 = (unsigned __int64 *)(v30 + 160 * v29);
          if ( (unsigned __int64 *)v30 != v31 )
          {
            do
            {
              v31 -= 20;
              if ( (unsigned __int64 *)*v31 != v31 + 2 )
                _libc_free(*v31);
            }
            while ( (unsigned __int64 *)v30 != v31 );
            v30 = *(_QWORD *)(v28 + 16);
          }
          if ( v30 != v28 + 32 )
            _libc_free(v30);
        }
        *(_DWORD *)(a1 + 8) = v56;
        v32 = *(_QWORD *)a2;
        v33 = *(_QWORD *)a2 + 672LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v33 )
        {
          do
          {
            v34 = *(unsigned int *)(v33 - 648);
            v35 = *(_QWORD *)(v33 - 656);
            v33 -= 672;
            v36 = (unsigned __int64 *)(v35 + 160 * v34);
            if ( (unsigned __int64 *)v35 != v36 )
            {
              do
              {
                v36 -= 20;
                if ( (unsigned __int64 *)*v36 != v36 + 2 )
                  _libc_free(*v36);
              }
              while ( (unsigned __int64 *)v35 != v36 );
              v35 = *(_QWORD *)(v33 + 16);
            }
            if ( v35 != v33 + 32 )
              _libc_free(v35);
          }
          while ( v32 != v33 );
        }
      }
      else
      {
        v12 = *(unsigned int *)(a1 + 12);
        if ( v11 > v12 )
        {
          v44 = 5 * v10;
          v45 = v62 + 672 * v10;
          while ( v62 != v45 )
          {
            v46 = *(unsigned int *)(v45 - 648);
            v47 = *(_QWORD *)(v45 - 656);
            v45 -= 672LL;
            v46 *= 160;
            v48 = (unsigned __int64 *)(v47 + v46);
            if ( v47 != v47 + v46 )
            {
              do
              {
                v48 -= 20;
                v44 = (__int64)(v48 + 2);
                if ( (unsigned __int64 *)*v48 != v48 + 2 )
                  _libc_free(*v48);
              }
              while ( (unsigned __int64 *)v47 != v48 );
              v47 = *(_QWORD *)(v45 + 16);
            }
            if ( v47 != v45 + 32 )
              _libc_free(v47);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_23672E0(a1, v11, v44, v6, a5, a6);
          v13 = *(_QWORD *)a2;
          v57 = *(unsigned int *)(a2 + 8);
          v10 = 0;
          v58 = *(_QWORD *)a2;
          v62 = *(_QWORD *)a1;
        }
        else
        {
          v13 = v6;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v12 = 5 * v10;
            v37 = v62 + 16;
            v60 = 672 * v10;
            v10 *= 672LL;
            v38 = a2 + 32;
            v39 = v62 + 16 + v10;
            do
            {
              v40 = *(_QWORD *)(v38 - 16);
              v41 = v38;
              v42 = v37;
              v63 = v10;
              v37 += 672;
              v38 += 672;
              *(_QWORD *)(v37 - 688) = v40;
              v43 = *(_QWORD *)(v38 - 680);
              *(_QWORD *)(v37 - 680) = v43;
              sub_2731860(v42, v41, v12, v43, a5, a6);
              v10 = v63;
            }
            while ( v37 != v39 );
            v57 = *(unsigned int *)(a2 + 8);
            v58 = *(_QWORD *)a2;
            v13 = *(_QWORD *)a2 + v60;
            v62 = *(_QWORD *)a1;
          }
        }
        v14 = v57;
        v15 = v10 + v62;
        for ( i = v58 + 672 * v57; i != v13; v15 += 672LL )
        {
          if ( v15 )
          {
            *(_QWORD *)v15 = *(_QWORD *)v13;
            v17 = *(_QWORD *)(v13 + 8);
            *(_DWORD *)(v15 + 24) = 0;
            *(_QWORD *)(v15 + 8) = v17;
            *(_QWORD *)(v15 + 16) = v15 + 32;
            *(_DWORD *)(v15 + 28) = 4;
            if ( *(_DWORD *)(v13 + 24) )
              sub_2731860(v15 + 16, v13 + 16, v12, v14, a5, a6);
          }
          v13 += 672;
        }
        *(_DWORD *)(a1 + 8) = v56;
        v18 = *(_QWORD *)a2;
        v19 = *(_QWORD *)a2 + 672LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v19 )
        {
          do
          {
            v20 = *(unsigned int *)(v19 - 648);
            v21 = *(_QWORD *)(v19 - 656);
            v19 -= 672;
            v22 = (unsigned __int64 *)(v21 + 160 * v20);
            if ( (unsigned __int64 *)v21 != v22 )
            {
              do
              {
                v22 -= 20;
                if ( (unsigned __int64 *)*v22 != v22 + 2 )
                  _libc_free(*v22);
              }
              while ( (unsigned __int64 *)v21 != v22 );
              v21 = *(_QWORD *)(v19 + 16);
            }
            if ( v21 != v19 + 32 )
              _libc_free(v21);
          }
          while ( v18 != v19 );
        }
      }
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v23 = v9 + 672 * v10;
      if ( v23 != v9 )
      {
        do
        {
          v24 = *(unsigned int *)(v23 - 648);
          v25 = *(_QWORD *)(v23 - 656);
          v23 -= 672LL;
          v26 = (unsigned __int64 *)(v25 + 160 * v24);
          if ( (unsigned __int64 *)v25 != v26 )
          {
            do
            {
              v26 -= 20;
              if ( (unsigned __int64 *)*v26 != v26 + 2 )
                _libc_free(*v26);
            }
            while ( (unsigned __int64 *)v25 != v26 );
            v25 = *(_QWORD *)(v23 + 16);
          }
          if ( v25 != v23 + 32 )
            _libc_free(v25);
        }
        while ( v23 != v62 );
        v59 = *(_QWORD *)a1;
      }
      if ( v59 != a1 + 16 )
        _libc_free(v59);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v58;
    }
  }
}
