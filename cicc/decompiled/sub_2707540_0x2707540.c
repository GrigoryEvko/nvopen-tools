// Function: sub_2707540
// Address: 0x2707540
//
void __fastcall sub_2707540(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rcx
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 v15; // r8
  __int64 v16; // r13
  __int64 v17; // r12
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // r15
  __int64 v20; // rbx
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rbx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // r15
  __int64 v25; // r12
  unsigned __int64 v26; // rdi
  unsigned __int64 v27; // rax
  unsigned __int64 v28; // rbx
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // r15
  __int64 v31; // r13
  unsigned __int64 v32; // rdi
  __int64 v33; // r13
  __int64 v34; // r12
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // r15
  __int64 v37; // rbx
  unsigned __int64 v38; // rdi
  unsigned __int64 v39; // r15
  unsigned __int64 v40; // rdx
  unsigned __int64 v41; // rbx
  __int64 v42; // rax
  unsigned __int64 v43; // rdi
  bool v44; // zf
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  __int64 v47; // rcx
  unsigned __int64 v48; // rbx
  unsigned __int64 v49; // rdi
  unsigned __int64 v50; // r13
  __int64 v51; // r12
  unsigned __int64 v52; // rdi
  unsigned __int64 v53; // r15
  unsigned __int64 v54; // rdx
  unsigned __int64 v55; // r15
  __int64 v56; // rbx
  unsigned __int64 v57; // rdi
  unsigned __int64 v58; // rdi
  unsigned __int64 v59; // rdi
  unsigned __int64 v60; // [rsp+8h] [rbp-58h]
  unsigned int v61; // [rsp+14h] [rbp-4Ch]
  __int64 v62; // [rsp+18h] [rbp-48h]
  unsigned __int64 v64; // [rsp+28h] [rbp-38h]
  __int64 v65; // [rsp+28h] [rbp-38h]
  __int64 v66; // [rsp+28h] [rbp-38h]
  __int64 v67; // [rsp+28h] [rbp-38h]
  __int64 v68; // [rsp+28h] [rbp-38h]
  unsigned __int64 v69; // [rsp+28h] [rbp-38h]

  if ( a1 == a2 )
    return;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v8 = a2 + 16;
  v62 = v6;
  v64 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 == a2 + 16 )
  {
    v61 = *(_DWORD *)(a2 + 8);
    v9 = v61;
    if ( v61 > v6 )
    {
      if ( v61 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v47 = *(_QWORD *)a1;
        v48 = v64 + 40 * v62;
        while ( v48 != v64 )
        {
          v48 -= 40LL;
          v49 = *(_QWORD *)(v48 + 16);
          if ( v49 != v48 + 40 )
            _libc_free(v49);
          v50 = *(_QWORD *)v48;
          v51 = *(_QWORD *)v48 + 80LL * *(unsigned int *)(v48 + 8);
          if ( *(_QWORD *)v48 != v51 )
          {
            do
            {
              v51 -= 80;
              v52 = *(_QWORD *)(v51 + 8);
              if ( v52 != v51 + 24 )
                _libc_free(v52);
            }
            while ( v50 != v51 );
            v50 = *(_QWORD *)v48;
          }
          if ( v50 != v48 + 16 )
            _libc_free(v50);
        }
        *(_DWORD *)(a1 + 8) = 0;
        sub_F31630(a1, v61, a3, v47, a5, a6);
        v8 = *(_QWORD *)a2;
        v62 = 0;
        v9 = *(unsigned int *)(a2 + 8);
        v10 = *(_QWORD *)a2;
        v64 = *(_QWORD *)a1;
LABEL_6:
        v11 = v10;
        v12 = v62 + v64;
        v13 = v8 + 40 * v9;
        if ( v13 != v10 )
        {
          do
          {
            v14 = 40;
            if ( v12 )
            {
              v15 = v12 + 16;
              *(_DWORD *)(v12 + 8) = 0;
              *(_QWORD *)v12 = v12 + 16;
              *(_DWORD *)(v12 + 12) = 0;
              if ( *(_DWORD *)(v11 + 8) )
              {
                v65 = v13;
                sub_2706EB0(v12, v11, v13, v6, v15, a6);
                v15 = v12 + 16;
                v13 = v65;
              }
              v14 = v12 + 40;
              *(_QWORD *)(v12 + 24) = 0;
              *(_QWORD *)(v12 + 16) = v12 + 40;
              *(_QWORD *)(v12 + 32) = 0;
              if ( *(_QWORD *)(v11 + 24) )
              {
                v66 = v13;
                sub_26F64C0(v15, (char **)(v11 + 16), v13, v6, v15, a6);
                v13 = v66;
              }
            }
            v11 += 40;
            v12 = v14;
          }
          while ( v13 != v11 );
        }
        *(_DWORD *)(a1 + 8) = v61;
        v16 = *(_QWORD *)a2;
        v17 = *(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v17 )
        {
          do
          {
            v17 -= 40;
            v18 = *(_QWORD *)(v17 + 16);
            if ( v18 != v17 + 40 )
              _libc_free(v18);
            v19 = *(_QWORD *)v17;
            v20 = *(_QWORD *)v17 + 80LL * *(unsigned int *)(v17 + 8);
            if ( *(_QWORD *)v17 != v20 )
            {
              do
              {
                v20 -= 80;
                v21 = *(_QWORD *)(v20 + 8);
                if ( v21 != v20 + 24 )
                  _libc_free(v21);
              }
              while ( v19 != v20 );
              v19 = *(_QWORD *)v17;
            }
            if ( v19 != v17 + 16 )
              _libc_free(v19);
          }
          while ( v16 != v17 );
        }
        goto LABEL_25;
      }
      v10 = a2 + 16;
      if ( !*(_DWORD *)(a1 + 8) )
        goto LABEL_6;
      v62 = 40 * v6;
      v60 = v7 + 40 * v6;
      while ( 1 )
      {
        v39 = v7 + 40;
        v40 = v7 + 40;
        if ( v7 != v8 )
        {
          v41 = *(_QWORD *)v7;
          v42 = *(_QWORD *)v7 + 80LL * *(unsigned int *)(v7 + 8);
          if ( *(_DWORD *)(v8 + 8) )
          {
            if ( v42 != v41 )
            {
              do
              {
                v42 -= 80;
                v43 = *(_QWORD *)(v42 + 8);
                if ( v43 != v42 + 24 )
                {
                  v67 = v42;
                  _libc_free(v43);
                  v42 = v67;
                }
              }
              while ( v42 != v41 );
              v41 = *(_QWORD *)v7;
            }
            if ( v41 != v7 + 16 )
              _libc_free(v41);
            *(_QWORD *)v7 = *(_QWORD *)v8;
            *(_DWORD *)(v7 + 8) = *(_DWORD *)(v8 + 8);
            *(_DWORD *)(v7 + 12) = *(_DWORD *)(v8 + 12);
            v44 = *(_QWORD *)(v8 + 24) == 0;
            *(_QWORD *)v8 = v8 + 16;
            *(_DWORD *)(v8 + 12) = 0;
            *(_DWORD *)(v8 + 8) = 0;
            if ( !v44 )
              goto LABEL_84;
          }
          else
          {
            while ( v42 != v41 )
            {
              while ( 1 )
              {
                v42 -= 80;
                v45 = *(_QWORD *)(v42 + 8);
                if ( v45 == v42 + 24 )
                  break;
                v68 = v42;
                _libc_free(v45);
                v42 = v68;
                if ( v68 == v41 )
                  goto LABEL_83;
              }
            }
LABEL_83:
            *(_DWORD *)(v7 + 8) = 0;
            if ( *(_QWORD *)(v8 + 24) )
            {
LABEL_84:
              v46 = *(_QWORD *)(v7 + 16);
              v39 = v7 + 40;
              if ( v46 != v7 + 40 )
                _libc_free(v46);
              *(_QWORD *)(v7 + 16) = *(_QWORD *)(v8 + 16);
              *(_QWORD *)(v7 + 24) = *(_QWORD *)(v8 + 24);
              *(_QWORD *)(v7 + 32) = *(_QWORD *)(v8 + 32);
              v40 = v8 + 40;
              *(_QWORD *)(v8 + 16) = v8 + 40;
              *(_QWORD *)(v8 + 32) = 0;
              *(_QWORD *)(v8 + 24) = 0;
              goto LABEL_78;
            }
          }
          *(_QWORD *)(v7 + 24) = 0;
          v39 = v7 + 40;
          v40 = v8 + 40;
        }
LABEL_78:
        v8 = v40;
        v7 = v39;
        if ( v60 == v39 )
        {
          v8 = *(_QWORD *)a2;
          v6 = *(_QWORD *)a1;
          v9 = *(unsigned int *)(a2 + 8);
          v64 = *(_QWORD *)a1;
          v10 = *(_QWORD *)a2 + v62;
          goto LABEL_6;
        }
      }
    }
    v27 = *(_QWORD *)a1;
    if ( !v61 )
      goto LABEL_43;
    v69 = v7 + 40LL * v61;
    while ( 1 )
    {
      v53 = v7 + 40;
      v54 = v7 + 40;
      if ( v7 != v8 )
      {
        v55 = *(_QWORD *)v7;
        v56 = *(_QWORD *)v7 + 80LL * *(unsigned int *)(v7 + 8);
        if ( *(_DWORD *)(v8 + 8) )
        {
          if ( v56 != v55 )
          {
            do
            {
              v56 -= 80;
              v57 = *(_QWORD *)(v56 + 8);
              if ( v57 != v56 + 24 )
                _libc_free(v57);
            }
            while ( v56 != v55 );
            v55 = *(_QWORD *)v7;
          }
          if ( v55 != v7 + 16 )
            _libc_free(v55);
          *(_QWORD *)v7 = *(_QWORD *)v8;
          *(_DWORD *)(v7 + 8) = *(_DWORD *)(v8 + 8);
          *(_DWORD *)(v7 + 12) = *(_DWORD *)(v8 + 12);
          v44 = *(_QWORD *)(v8 + 24) == 0;
          *(_QWORD *)v8 = v8 + 16;
          *(_DWORD *)(v8 + 12) = 0;
          *(_DWORD *)(v8 + 8) = 0;
          if ( !v44 )
            goto LABEL_117;
        }
        else
        {
          while ( v56 != v55 )
          {
            while ( 1 )
            {
              v56 -= 80;
              v58 = *(_QWORD *)(v56 + 8);
              if ( v58 == v56 + 24 )
                break;
              _libc_free(v58);
              if ( v56 == v55 )
                goto LABEL_116;
            }
          }
LABEL_116:
          *(_DWORD *)(v7 + 8) = 0;
          if ( *(_QWORD *)(v8 + 24) )
          {
LABEL_117:
            v59 = *(_QWORD *)(v7 + 16);
            v53 = v7 + 40;
            if ( v59 != v7 + 40 )
              _libc_free(v59);
            *(_QWORD *)(v7 + 16) = *(_QWORD *)(v8 + 16);
            *(_QWORD *)(v7 + 24) = *(_QWORD *)(v8 + 24);
            *(_QWORD *)(v7 + 32) = *(_QWORD *)(v8 + 32);
            v54 = v8 + 40;
            *(_QWORD *)(v8 + 16) = v8 + 40;
            *(_QWORD *)(v8 + 32) = 0;
            *(_QWORD *)(v8 + 24) = 0;
            goto LABEL_111;
          }
        }
        *(_QWORD *)(v7 + 24) = 0;
        v53 = v7 + 40;
        v54 = v8 + 40;
      }
LABEL_111:
      v8 = v54;
      v7 = v53;
      if ( v53 == v69 )
      {
        v27 = *(_QWORD *)a1;
        v62 = *(unsigned int *)(a1 + 8);
LABEL_43:
        v28 = v27 + 40 * v62;
        while ( v7 != v28 )
        {
          v28 -= 40LL;
          v29 = *(_QWORD *)(v28 + 16);
          if ( v29 != v28 + 40 )
            _libc_free(v29);
          v30 = *(_QWORD *)v28;
          v31 = *(_QWORD *)v28 + 80LL * *(unsigned int *)(v28 + 8);
          if ( *(_QWORD *)v28 != v31 )
          {
            do
            {
              v31 -= 80;
              v32 = *(_QWORD *)(v31 + 8);
              if ( v32 != v31 + 24 )
                _libc_free(v32);
            }
            while ( v30 != v31 );
            v30 = *(_QWORD *)v28;
          }
          if ( v30 != v28 + 16 )
            _libc_free(v30);
        }
        *(_DWORD *)(a1 + 8) = v61;
        v33 = *(_QWORD *)a2;
        v34 = *(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v34 )
        {
          do
          {
            v34 -= 40;
            v35 = *(_QWORD *)(v34 + 16);
            if ( v35 != v34 + 40 )
              _libc_free(v35);
            v36 = *(_QWORD *)v34;
            v37 = *(_QWORD *)v34 + 80LL * *(unsigned int *)(v34 + 8);
            if ( *(_QWORD *)v34 != v37 )
            {
              do
              {
                v37 -= 80;
                v38 = *(_QWORD *)(v37 + 8);
                if ( v38 != v37 + 24 )
                  _libc_free(v38);
              }
              while ( v36 != v37 );
              v36 = *(_QWORD *)v34;
            }
            if ( v36 != v34 + 16 )
              _libc_free(v36);
          }
          while ( v33 != v34 );
          *(_DWORD *)(a2 + 8) = 0;
          return;
        }
LABEL_25:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
    }
  }
  v22 = v7 + 40 * v6;
  if ( v22 != v7 )
  {
    do
    {
      v22 -= 40LL;
      v23 = *(_QWORD *)(v22 + 16);
      if ( v23 != v22 + 40 )
        _libc_free(v23);
      v24 = *(_QWORD *)v22;
      v25 = *(_QWORD *)v22 + 80LL * *(unsigned int *)(v22 + 8);
      if ( *(_QWORD *)v22 != v25 )
      {
        do
        {
          v25 -= 80;
          v26 = *(_QWORD *)(v25 + 8);
          if ( v26 != v25 + 24 )
            _libc_free(v26);
        }
        while ( v24 != v25 );
        v24 = *(_QWORD *)v22;
      }
      if ( v24 != v22 + 16 )
        _libc_free(v24);
    }
    while ( v22 != v64 );
    v7 = *(_QWORD *)a1;
  }
  if ( v7 != a1 + 16 )
    _libc_free(v7);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_QWORD *)a2 = v8;
  *(_QWORD *)(a2 + 8) = 0;
}
