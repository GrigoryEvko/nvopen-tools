// Function: sub_109AD70
// Address: 0x109ad70
//
void __fastcall sub_109AD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rcx
  __int64 v7; // r12
  __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // rdx
  __int64 v15; // r13
  __int64 v16; // r8
  __int64 v17; // r13
  __int64 v18; // r12
  __int64 v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rbx
  __int64 v22; // rdi
  unsigned __int64 v23; // rbx
  __int64 v24; // rdi
  __int64 v25; // r15
  __int64 v26; // r12
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rdi
  __int64 v31; // r15
  __int64 v32; // r13
  __int64 v33; // rdi
  __int64 v34; // r13
  __int64 v35; // r12
  __int64 v36; // rdi
  __int64 v37; // r15
  __int64 v38; // rbx
  __int64 v39; // rdi
  __int64 v40; // r15
  __int64 v41; // rdx
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rdi
  bool v45; // zf
  __int64 v46; // rdi
  __int64 v47; // rdi
  __int64 v48; // rcx
  __int64 v49; // rbx
  __int64 v50; // rdi
  __int64 v51; // r13
  __int64 v52; // r12
  __int64 v53; // rdi
  __int64 v54; // r15
  __int64 v55; // rdx
  __int64 v56; // r15
  __int64 v57; // rbx
  __int64 v58; // rdi
  __int64 v59; // rdi
  __int64 v60; // rdi
  unsigned __int64 v61; // [rsp+8h] [rbp-58h]
  unsigned int v62; // [rsp+14h] [rbp-4Ch]
  __int64 v63; // [rsp+18h] [rbp-48h]
  __int64 v65; // [rsp+28h] [rbp-38h]
  __int64 v66; // [rsp+28h] [rbp-38h]
  __int64 v67; // [rsp+28h] [rbp-38h]
  __int64 v68; // [rsp+28h] [rbp-38h]
  __int64 v69; // [rsp+28h] [rbp-38h]
  __int64 v70; // [rsp+28h] [rbp-38h]

  if ( a1 == a2 )
    return;
  v6 = *(unsigned int *)(a1 + 8);
  v7 = *(_QWORD *)a1;
  v8 = a2;
  v9 = a2 + 16;
  v63 = v6;
  v65 = *(_QWORD *)a1;
  if ( *(_QWORD *)a2 == a2 + 16 )
  {
    v62 = *(_DWORD *)(a2 + 8);
    v10 = v62;
    if ( v62 > v6 )
    {
      if ( v62 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
      {
        v48 = *(_QWORD *)a1;
        v49 = v65 + 40 * v63;
        while ( v49 != v65 )
        {
          v49 -= 40;
          v50 = *(_QWORD *)(v49 + 16);
          if ( v50 != v49 + 40 )
            _libc_free(v50, a2);
          v51 = *(_QWORD *)v49;
          v52 = *(_QWORD *)v49 + 80LL * *(unsigned int *)(v49 + 8);
          if ( *(_QWORD *)v49 != v52 )
          {
            do
            {
              v52 -= 80;
              v53 = *(_QWORD *)(v52 + 8);
              if ( v53 != v52 + 24 )
                _libc_free(v53, a2);
            }
            while ( v51 != v52 );
            v51 = *(_QWORD *)v49;
          }
          if ( v51 != v49 + 16 )
            _libc_free(v51, a2);
        }
        *(_DWORD *)(a1 + 8) = 0;
        sub_F31630(a1, v62, a3, v48, a5, a6);
        v9 = *(_QWORD *)a2;
        v63 = 0;
        v10 = *(unsigned int *)(a2 + 8);
        v11 = *(_QWORD *)a2;
        v65 = *(_QWORD *)a1;
LABEL_6:
        v12 = v11;
        v13 = v63 + v65;
        v14 = v9 + 40 * v10;
        if ( v14 != v11 )
        {
          do
          {
            v15 = 40;
            if ( v13 )
            {
              v16 = v13 + 16;
              *(_DWORD *)(v13 + 8) = 0;
              *(_QWORD *)v13 = v13 + 16;
              *(_DWORD *)(v13 + 12) = 0;
              if ( *(_DWORD *)(v12 + 8) )
              {
                v66 = v14;
                sub_109A3D0(v13, v12, v14, v6, v16, a6);
                v16 = v13 + 16;
                v14 = v66;
              }
              v15 = v13 + 40;
              *(_QWORD *)(v13 + 24) = 0;
              *(_QWORD *)(v13 + 16) = v13 + 40;
              *(_QWORD *)(v13 + 32) = 0;
              if ( *(_QWORD *)(v12 + 24) )
              {
                v67 = v14;
                sub_1099150(v16, (char **)(v12 + 16), v14, v6, v16, a6);
                v14 = v67;
              }
            }
            v12 += 40;
            v13 = v15;
          }
          while ( v14 != v12 );
        }
        *(_DWORD *)(a1 + 8) = v62;
        v17 = *(_QWORD *)a2;
        v18 = *(_QWORD *)a2 + 40LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v18 )
        {
          do
          {
            v18 -= 40;
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 != v18 + 40 )
              _libc_free(v19, v62);
            v20 = *(_QWORD *)v18;
            v21 = *(_QWORD *)v18 + 80LL * *(unsigned int *)(v18 + 8);
            if ( *(_QWORD *)v18 != v21 )
            {
              do
              {
                v21 -= 80;
                v22 = *(_QWORD *)(v21 + 8);
                if ( v22 != v21 + 24 )
                  _libc_free(v22, v62);
              }
              while ( v20 != v21 );
              v20 = *(_QWORD *)v18;
            }
            if ( v20 != v18 + 16 )
              _libc_free(v20, v62);
          }
          while ( v17 != v18 );
        }
        goto LABEL_25;
      }
      v11 = a2 + 16;
      if ( !*(_DWORD *)(a1 + 8) )
        goto LABEL_6;
      v63 = 40 * v6;
      v61 = v7 + 40 * v6;
      while ( 1 )
      {
        v40 = v7 + 40;
        v41 = v7 + 40;
        if ( v7 != v9 )
        {
          v42 = *(_QWORD *)v7;
          v43 = *(_QWORD *)v7 + 80LL * *(unsigned int *)(v7 + 8);
          if ( *(_DWORD *)(v9 + 8) )
          {
            if ( v43 != v42 )
            {
              do
              {
                v43 -= 80;
                v44 = *(_QWORD *)(v43 + 8);
                if ( v44 != v43 + 24 )
                {
                  v68 = v43;
                  _libc_free(v44, a2);
                  v43 = v68;
                }
              }
              while ( v43 != v42 );
              v42 = *(_QWORD *)v7;
            }
            if ( v42 != v7 + 16 )
              _libc_free(v42, a2);
            *(_QWORD *)v7 = *(_QWORD *)v9;
            *(_DWORD *)(v7 + 8) = *(_DWORD *)(v9 + 8);
            *(_DWORD *)(v7 + 12) = *(_DWORD *)(v9 + 12);
            v45 = *(_QWORD *)(v9 + 24) == 0;
            *(_QWORD *)v9 = v9 + 16;
            *(_DWORD *)(v9 + 12) = 0;
            *(_DWORD *)(v9 + 8) = 0;
            if ( !v45 )
              goto LABEL_84;
          }
          else
          {
            while ( v43 != v42 )
            {
              while ( 1 )
              {
                v43 -= 80;
                v46 = *(_QWORD *)(v43 + 8);
                if ( v46 == v43 + 24 )
                  break;
                v69 = v43;
                _libc_free(v46, a2);
                v43 = v69;
                if ( v69 == v42 )
                  goto LABEL_83;
              }
            }
LABEL_83:
            *(_DWORD *)(v7 + 8) = 0;
            if ( *(_QWORD *)(v9 + 24) )
            {
LABEL_84:
              v47 = *(_QWORD *)(v7 + 16);
              v40 = v7 + 40;
              if ( v47 != v7 + 40 )
                _libc_free(v47, a2);
              *(_QWORD *)(v7 + 16) = *(_QWORD *)(v9 + 16);
              *(_QWORD *)(v7 + 24) = *(_QWORD *)(v9 + 24);
              *(_QWORD *)(v7 + 32) = *(_QWORD *)(v9 + 32);
              v41 = v9 + 40;
              *(_QWORD *)(v9 + 16) = v9 + 40;
              *(_QWORD *)(v9 + 32) = 0;
              *(_QWORD *)(v9 + 24) = 0;
              goto LABEL_78;
            }
          }
          *(_QWORD *)(v7 + 24) = 0;
          v40 = v7 + 40;
          v41 = v9 + 40;
        }
LABEL_78:
        v9 = v41;
        v7 = v40;
        if ( v61 == v40 )
        {
          v9 = *(_QWORD *)a2;
          v6 = *(_QWORD *)a1;
          v10 = *(unsigned int *)(a2 + 8);
          v65 = *(_QWORD *)a1;
          v11 = *(_QWORD *)a2 + v63;
          goto LABEL_6;
        }
      }
    }
    v28 = *(_QWORD *)a1;
    if ( !v62 )
      goto LABEL_43;
    v70 = v7 + 40LL * v62;
    while ( 1 )
    {
      v54 = v7 + 40;
      v55 = v7 + 40;
      if ( v7 != v9 )
      {
        v56 = *(_QWORD *)v7;
        v57 = *(_QWORD *)v7 + 80LL * *(unsigned int *)(v7 + 8);
        if ( *(_DWORD *)(v9 + 8) )
        {
          if ( v57 != v56 )
          {
            do
            {
              v57 -= 80;
              v58 = *(_QWORD *)(v57 + 8);
              if ( v58 != v57 + 24 )
                _libc_free(v58, a2);
            }
            while ( v57 != v56 );
            v56 = *(_QWORD *)v7;
          }
          if ( v56 != v7 + 16 )
            _libc_free(v56, a2);
          *(_QWORD *)v7 = *(_QWORD *)v9;
          *(_DWORD *)(v7 + 8) = *(_DWORD *)(v9 + 8);
          *(_DWORD *)(v7 + 12) = *(_DWORD *)(v9 + 12);
          v45 = *(_QWORD *)(v9 + 24) == 0;
          *(_QWORD *)v9 = v9 + 16;
          *(_DWORD *)(v9 + 12) = 0;
          *(_DWORD *)(v9 + 8) = 0;
          if ( !v45 )
            goto LABEL_117;
        }
        else
        {
          while ( v57 != v56 )
          {
            while ( 1 )
            {
              v57 -= 80;
              v59 = *(_QWORD *)(v57 + 8);
              if ( v59 == v57 + 24 )
                break;
              _libc_free(v59, a2);
              if ( v57 == v56 )
                goto LABEL_116;
            }
          }
LABEL_116:
          *(_DWORD *)(v7 + 8) = 0;
          if ( *(_QWORD *)(v9 + 24) )
          {
LABEL_117:
            v60 = *(_QWORD *)(v7 + 16);
            v54 = v7 + 40;
            if ( v60 != v7 + 40 )
              _libc_free(v60, a2);
            *(_QWORD *)(v7 + 16) = *(_QWORD *)(v9 + 16);
            *(_QWORD *)(v7 + 24) = *(_QWORD *)(v9 + 24);
            *(_QWORD *)(v7 + 32) = *(_QWORD *)(v9 + 32);
            v55 = v9 + 40;
            *(_QWORD *)(v9 + 16) = v9 + 40;
            *(_QWORD *)(v9 + 32) = 0;
            *(_QWORD *)(v9 + 24) = 0;
            goto LABEL_111;
          }
        }
        *(_QWORD *)(v7 + 24) = 0;
        v54 = v7 + 40;
        v55 = v9 + 40;
      }
LABEL_111:
      v9 = v55;
      v7 = v54;
      if ( v54 == v70 )
      {
        v28 = *(_QWORD *)a1;
        a2 = *(unsigned int *)(a1 + 8);
        v63 = a2;
LABEL_43:
        v29 = v28 + 40 * v63;
        while ( v7 != v29 )
        {
          v29 -= 40;
          v30 = *(_QWORD *)(v29 + 16);
          if ( v30 != v29 + 40 )
            _libc_free(v30, a2);
          v31 = *(_QWORD *)v29;
          v32 = *(_QWORD *)v29 + 80LL * *(unsigned int *)(v29 + 8);
          if ( *(_QWORD *)v29 != v32 )
          {
            do
            {
              v32 -= 80;
              v33 = *(_QWORD *)(v32 + 8);
              if ( v33 != v32 + 24 )
                _libc_free(v33, a2);
            }
            while ( v31 != v32 );
            v31 = *(_QWORD *)v29;
          }
          if ( v31 != v29 + 16 )
            _libc_free(v31, a2);
        }
        *(_DWORD *)(a1 + 8) = v62;
        v34 = *(_QWORD *)v8;
        v35 = *(_QWORD *)v8 + 40LL * *(unsigned int *)(v8 + 8);
        if ( *(_QWORD *)v8 != v35 )
        {
          do
          {
            v35 -= 40;
            v36 = *(_QWORD *)(v35 + 16);
            if ( v36 != v35 + 40 )
              _libc_free(v36, v62);
            v37 = *(_QWORD *)v35;
            v38 = *(_QWORD *)v35 + 80LL * *(unsigned int *)(v35 + 8);
            if ( *(_QWORD *)v35 != v38 )
            {
              do
              {
                v38 -= 80;
                v39 = *(_QWORD *)(v38 + 8);
                if ( v39 != v38 + 24 )
                  _libc_free(v39, v62);
              }
              while ( v37 != v38 );
              v37 = *(_QWORD *)v35;
            }
            if ( v37 != v35 + 16 )
              _libc_free(v37, v62);
          }
          while ( v34 != v35 );
          *(_DWORD *)(v8 + 8) = 0;
          return;
        }
LABEL_25:
        *(_DWORD *)(v8 + 8) = 0;
        return;
      }
    }
  }
  v23 = v7 + 40 * v6;
  if ( v23 != v7 )
  {
    do
    {
      v23 -= 40LL;
      v24 = *(_QWORD *)(v23 + 16);
      if ( v24 != v23 + 40 )
        _libc_free(v24, a2);
      v25 = *(_QWORD *)v23;
      v26 = *(_QWORD *)v23 + 80LL * *(unsigned int *)(v23 + 8);
      if ( *(_QWORD *)v23 != v26 )
      {
        do
        {
          v26 -= 80;
          v27 = *(_QWORD *)(v26 + 8);
          if ( v27 != v26 + 24 )
            _libc_free(v27, a2);
        }
        while ( v25 != v26 );
        v25 = *(_QWORD *)v23;
      }
      if ( v25 != v23 + 16 )
        _libc_free(v25, a2);
    }
    while ( v23 != v65 );
    v7 = *(_QWORD *)a1;
  }
  if ( v7 != a1 + 16 )
    _libc_free(v7, a2);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
  *(_QWORD *)a2 = v9;
  *(_QWORD *)(a2 + 8) = 0;
}
