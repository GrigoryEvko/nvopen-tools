// Function: sub_15ED230
// Address: 0x15ed230
//
void __fastcall sub_15ED230(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // rcx
  unsigned __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rcx
  __int64 v7; // rbx
  unsigned __int64 v8; // r12
  __int64 i; // r13
  int v10; // eax
  __int64 v11; // rdx
  __int64 v12; // r14
  __int64 v13; // rdx
  unsigned __int64 v14; // r13
  unsigned __int64 v15; // rbx
  __int64 v16; // rax
  unsigned __int64 v17; // r12
  _QWORD *v18; // r15
  unsigned __int64 v19; // r12
  _QWORD *v20; // rbx
  unsigned __int64 v21; // r14
  __int64 v22; // rdx
  unsigned __int64 v23; // r13
  unsigned __int64 v24; // rbx
  __int64 v25; // rax
  unsigned __int64 v26; // r12
  _QWORD *v27; // r15
  unsigned __int64 v28; // r12
  _QWORD *v29; // rbx
  unsigned __int64 v30; // r13
  unsigned __int64 v31; // r13
  __int64 v32; // rdx
  unsigned __int64 v33; // r12
  unsigned __int64 v34; // rbx
  __int64 v35; // r14
  unsigned __int64 v36; // r15
  _QWORD *v37; // r14
  unsigned __int64 v38; // r12
  _QWORD *v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rdx
  unsigned __int64 v42; // r12
  unsigned __int64 v43; // rbx
  __int64 v44; // rax
  unsigned __int64 v45; // r15
  _QWORD *v46; // r14
  unsigned __int64 v47; // r12
  _QWORD *v48; // rbx
  __int64 v49; // rcx
  __int64 v50; // rbx
  __int64 v51; // r14
  __int64 v52; // r12
  __int64 v53; // r13
  __int64 v54; // rdx
  __int64 v55; // rdi
  unsigned __int64 v56; // r13
  __int64 v57; // rdx
  unsigned __int64 v58; // r12
  unsigned __int64 v59; // rbx
  __int64 v60; // rax
  unsigned __int64 v61; // r15
  _QWORD *v62; // r14
  unsigned __int64 v63; // r12
  _QWORD *v64; // rbx
  __int64 v65; // r13
  __int64 v66; // r14
  __int64 v67; // rdi
  unsigned int v68; // [rsp+Ch] [rbp-64h]
  __int64 v69; // [rsp+10h] [rbp-60h]
  __int64 v70; // [rsp+18h] [rbp-58h]
  unsigned __int64 v72; // [rsp+30h] [rbp-40h]
  unsigned __int64 v73; // [rsp+38h] [rbp-38h]
  __int64 v74; // [rsp+38h] [rbp-38h]
  __int64 v75; // [rsp+38h] [rbp-38h]
  unsigned __int64 v76; // [rsp+38h] [rbp-38h]

  if ( a1 != a2 )
  {
    v3 = *(_QWORD *)a1;
    v4 = *(unsigned int *)(a1 + 8);
    v70 = a2 + 16;
    v73 = *(_QWORD *)a1;
    v72 = *(_QWORD *)a1;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v68 = *(_DWORD *)(a2 + 8);
      v5 = v68;
      v69 = v68;
      if ( v68 > v4 )
      {
        v6 = a1;
        if ( v68 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          v56 = v73 + 192 * v4;
          while ( v56 != v73 )
          {
            v57 = *(unsigned int *)(v56 - 120);
            v58 = *(_QWORD *)(v56 - 128);
            v56 -= 192LL;
            v59 = v58 + 56 * v57;
            if ( v58 != v59 )
            {
              do
              {
                v60 = *(unsigned int *)(v59 - 40);
                v61 = *(_QWORD *)(v59 - 48);
                v59 -= 56LL;
                v60 *= 32;
                v62 = (_QWORD *)(v61 + v60);
                if ( v61 != v61 + v60 )
                {
                  do
                  {
                    v62 -= 4;
                    if ( (_QWORD *)*v62 != v62 + 2 )
                      j_j___libc_free_0(*v62, v62[2] + 1LL);
                  }
                  while ( (_QWORD *)v61 != v62 );
                  v61 = *(_QWORD *)(v59 + 8);
                }
                if ( v61 != v59 + 24 )
                  _libc_free(v61);
              }
              while ( v58 != v59 );
              v58 = *(_QWORD *)(v56 + 64);
            }
            if ( v58 != v56 + 80 )
              _libc_free(v58);
            v63 = *(_QWORD *)(v56 + 16);
            v64 = (_QWORD *)(v63 + 32LL * *(unsigned int *)(v56 + 24));
            if ( (_QWORD *)v63 != v64 )
            {
              do
              {
                v64 -= 4;
                if ( (_QWORD *)*v64 != v64 + 2 )
                  j_j___libc_free_0(*v64, v64[2] + 1LL);
              }
              while ( (_QWORD *)v63 != v64 );
              v63 = *(_QWORD *)(v56 + 16);
            }
            if ( v63 != v56 + 32 )
              _libc_free(v63);
          }
          *(_DWORD *)(a1 + 8) = 0;
          sub_15ECA10(a1, v68);
          v6 = *(_QWORD *)a1;
          v7 = *(_QWORD *)a2;
          v73 = *(_QWORD *)a1;
          v69 = *(unsigned int *)(a2 + 8);
          v4 = 0;
          v70 = *(_QWORD *)a2;
        }
        else
        {
          v7 = a2 + 16;
          if ( *(_DWORD *)(a1 + 8) )
          {
            v49 = a2;
            v50 = 192 * v4;
            v51 = v73 + 16;
            v4 = v50;
            v52 = a2 + 32;
            v53 = v73 + 16 + v50;
            do
            {
              v76 = v4;
              *(_DWORD *)(v51 - 16) = *(_DWORD *)(v52 - 16);
              *(_DWORD *)(v51 - 12) = *(_DWORD *)(v52 - 12);
              *(_BYTE *)(v51 - 8) = *(_BYTE *)(v52 - 8);
              *(_BYTE *)(v51 - 7) = *(_BYTE *)(v52 - 7);
              *(_BYTE *)(v51 - 6) = *(_BYTE *)(v52 - 6);
              *(_BYTE *)(v51 - 5) = *(_BYTE *)(v52 - 5);
              v54 = *(unsigned int *)(v52 - 4);
              *(_DWORD *)(v51 - 4) = v54;
              sub_15EB170(v51, v52, v54, v49);
              v55 = v51 + 48;
              v51 += 192;
              sub_15EC260(v55, v52 + 48);
              v52 += 192;
              v4 = v76;
            }
            while ( v51 != v53 );
            v69 = *(unsigned int *)(a2 + 8);
            v7 = *(_QWORD *)a2 + v50;
            v70 = *(_QWORD *)a2;
            v6 = *(_QWORD *)a1;
            v73 = *(_QWORD *)a1;
          }
        }
        v8 = v4 + v73;
        for ( i = v70 + 192 * v69; i != v7; v8 += 192LL )
        {
          if ( v8 )
          {
            *(_DWORD *)v8 = *(_DWORD *)v7;
            *(_DWORD *)(v8 + 4) = *(_DWORD *)(v7 + 4);
            *(_BYTE *)(v8 + 8) = *(_BYTE *)(v7 + 8);
            *(_BYTE *)(v8 + 9) = *(_BYTE *)(v7 + 9);
            *(_BYTE *)(v8 + 10) = *(_BYTE *)(v7 + 10);
            *(_BYTE *)(v8 + 11) = *(_BYTE *)(v7 + 11);
            v10 = *(_DWORD *)(v7 + 12);
            *(_DWORD *)(v8 + 24) = 0;
            *(_DWORD *)(v8 + 12) = v10;
            *(_QWORD *)(v8 + 16) = v8 + 32;
            *(_DWORD *)(v8 + 28) = 1;
            v11 = *(unsigned int *)(v7 + 24);
            if ( (_DWORD)v11 )
              sub_15EB170(v8 + 16, v7 + 16, v11, v6);
            *(_DWORD *)(v8 + 72) = 0;
            *(_QWORD *)(v8 + 64) = v8 + 80;
            *(_DWORD *)(v8 + 76) = 2;
            if ( *(_DWORD *)(v7 + 72) )
              sub_15EC260(v8 + 64, v7 + 64);
          }
          v7 += 192;
        }
        *(_DWORD *)(a1 + 8) = v68;
        v74 = *(_QWORD *)a2;
        v12 = *(_QWORD *)a2 + 192LL * *(unsigned int *)(a2 + 8);
        if ( *(_QWORD *)a2 != v12 )
        {
          do
          {
            v13 = *(unsigned int *)(v12 - 120);
            v14 = *(_QWORD *)(v12 - 128);
            v12 -= 192;
            v15 = v14 + 56 * v13;
            if ( v14 != v15 )
            {
              do
              {
                v16 = *(unsigned int *)(v15 - 40);
                v17 = *(_QWORD *)(v15 - 48);
                v15 -= 56LL;
                v16 *= 32;
                v18 = (_QWORD *)(v17 + v16);
                if ( v17 != v17 + v16 )
                {
                  do
                  {
                    v18 -= 4;
                    if ( (_QWORD *)*v18 != v18 + 2 )
                      j_j___libc_free_0(*v18, v18[2] + 1LL);
                  }
                  while ( (_QWORD *)v17 != v18 );
                  v17 = *(_QWORD *)(v15 + 8);
                }
                if ( v17 != v15 + 24 )
                  _libc_free(v17);
              }
              while ( v14 != v15 );
              v14 = *(_QWORD *)(v12 + 64);
            }
            if ( v14 != v12 + 80 )
              _libc_free(v14);
            v19 = *(_QWORD *)(v12 + 16);
            v20 = (_QWORD *)(v19 + 32LL * *(unsigned int *)(v12 + 24));
            if ( (_QWORD *)v19 != v20 )
            {
              do
              {
                v20 -= 4;
                if ( (_QWORD *)*v20 != v20 + 2 )
                  j_j___libc_free_0(*v20, v20[2] + 1LL);
              }
              while ( (_QWORD *)v19 != v20 );
              v19 = *(_QWORD *)(v12 + 16);
            }
            if ( v19 != v12 + 32 )
              _libc_free(v19);
          }
          while ( v74 != v12 );
        }
LABEL_35:
        *(_DWORD *)(a2 + 8) = 0;
        return;
      }
      v30 = *(_QWORD *)a1;
      if ( v68 )
      {
        v65 = v73 + 16;
        v66 = a2 + 32;
        do
        {
          *(_DWORD *)(v65 - 16) = *(_DWORD *)(v66 - 16);
          *(_DWORD *)(v65 - 12) = *(_DWORD *)(v66 - 12);
          *(_BYTE *)(v65 - 8) = *(_BYTE *)(v66 - 8);
          *(_BYTE *)(v65 - 7) = *(_BYTE *)(v66 - 7);
          *(_BYTE *)(v65 - 6) = *(_BYTE *)(v66 - 6);
          *(_BYTE *)(v65 - 5) = *(_BYTE *)(v66 - 5);
          *(_DWORD *)(v65 - 4) = *(_DWORD *)(v66 - 4);
          sub_15EB170(v65, v66, a3, v5);
          v67 = v65 + 48;
          v65 += 192;
          sub_15EC260(v67, v66 + 48);
          v66 += 192;
        }
        while ( v73 + 16 + 192LL * v68 != v65 );
        v72 = v73 + 192LL * v68;
        v30 = *(_QWORD *)a1;
        v4 = *(unsigned int *)(a1 + 8);
      }
      v31 = 192 * v4 + v30;
      while ( v72 != v31 )
      {
        v32 = *(unsigned int *)(v31 - 120);
        v33 = *(_QWORD *)(v31 - 128);
        v31 -= 192LL;
        v34 = v33 + 56 * v32;
        if ( v33 != v34 )
        {
          do
          {
            v35 = *(unsigned int *)(v34 - 40);
            v36 = *(_QWORD *)(v34 - 48);
            v34 -= 56LL;
            v37 = (_QWORD *)(v36 + 32 * v35);
            if ( (_QWORD *)v36 != v37 )
            {
              do
              {
                v37 -= 4;
                if ( (_QWORD *)*v37 != v37 + 2 )
                  j_j___libc_free_0(*v37, v37[2] + 1LL);
              }
              while ( (_QWORD *)v36 != v37 );
              v36 = *(_QWORD *)(v34 + 8);
            }
            if ( v36 != v34 + 24 )
              _libc_free(v36);
          }
          while ( v33 != v34 );
          v33 = *(_QWORD *)(v31 + 64);
        }
        if ( v33 != v31 + 80 )
          _libc_free(v33);
        v38 = *(_QWORD *)(v31 + 16);
        v39 = (_QWORD *)(v38 + 32LL * *(unsigned int *)(v31 + 24));
        if ( (_QWORD *)v38 != v39 )
        {
          do
          {
            v39 -= 4;
            if ( (_QWORD *)*v39 != v39 + 2 )
              j_j___libc_free_0(*v39, v39[2] + 1LL);
          }
          while ( (_QWORD *)v38 != v39 );
          v38 = *(_QWORD *)(v31 + 16);
        }
        if ( v38 != v31 + 32 )
          _libc_free(v38);
      }
      *(_DWORD *)(a1 + 8) = v68;
      v75 = *(_QWORD *)a2;
      v40 = *(_QWORD *)a2 + 192LL * *(unsigned int *)(a2 + 8);
      if ( *(_QWORD *)a2 == v40 )
        goto LABEL_35;
      do
      {
        v41 = *(unsigned int *)(v40 - 120);
        v42 = *(_QWORD *)(v40 - 128);
        v40 -= 192;
        v43 = v42 + 56 * v41;
        if ( v42 != v43 )
        {
          do
          {
            v44 = *(unsigned int *)(v43 - 40);
            v45 = *(_QWORD *)(v43 - 48);
            v43 -= 56LL;
            v44 *= 32;
            v46 = (_QWORD *)(v45 + v44);
            if ( v45 != v45 + v44 )
            {
              do
              {
                v46 -= 4;
                if ( (_QWORD *)*v46 != v46 + 2 )
                  j_j___libc_free_0(*v46, v46[2] + 1LL);
              }
              while ( (_QWORD *)v45 != v46 );
              v45 = *(_QWORD *)(v43 + 8);
            }
            if ( v45 != v43 + 24 )
              _libc_free(v45);
          }
          while ( v42 != v43 );
          v42 = *(_QWORD *)(v40 + 64);
        }
        if ( v42 != v40 + 80 )
          _libc_free(v42);
        v47 = *(_QWORD *)(v40 + 16);
        v48 = (_QWORD *)(v47 + 32LL * *(unsigned int *)(v40 + 24));
        if ( (_QWORD *)v47 != v48 )
        {
          do
          {
            v48 -= 4;
            if ( (_QWORD *)*v48 != v48 + 2 )
              j_j___libc_free_0(*v48, v48[2] + 1LL);
          }
          while ( (_QWORD *)v47 != v48 );
          v47 = *(_QWORD *)(v40 + 16);
        }
        if ( v47 != v40 + 32 )
          _libc_free(v47);
      }
      while ( v75 != v40 );
      *(_DWORD *)(a2 + 8) = 0;
    }
    else
    {
      v21 = v3 + 192 * v4;
      if ( v21 != v3 )
      {
        do
        {
          v22 = *(unsigned int *)(v21 - 120);
          v23 = *(_QWORD *)(v21 - 128);
          v21 -= 192LL;
          v24 = v23 + 56 * v22;
          if ( v23 != v24 )
          {
            do
            {
              v25 = *(unsigned int *)(v24 - 40);
              v26 = *(_QWORD *)(v24 - 48);
              v24 -= 56LL;
              v25 *= 32;
              v27 = (_QWORD *)(v26 + v25);
              if ( v26 != v26 + v25 )
              {
                do
                {
                  v27 -= 4;
                  if ( (_QWORD *)*v27 != v27 + 2 )
                    j_j___libc_free_0(*v27, v27[2] + 1LL);
                }
                while ( (_QWORD *)v26 != v27 );
                v26 = *(_QWORD *)(v24 + 8);
              }
              if ( v26 != v24 + 24 )
                _libc_free(v26);
            }
            while ( v23 != v24 );
            v23 = *(_QWORD *)(v21 + 64);
          }
          if ( v23 != v21 + 80 )
            _libc_free(v23);
          v28 = *(_QWORD *)(v21 + 16);
          v29 = (_QWORD *)(v28 + 32LL * *(unsigned int *)(v21 + 24));
          if ( (_QWORD *)v28 != v29 )
          {
            do
            {
              v29 -= 4;
              if ( (_QWORD *)*v29 != v29 + 2 )
                j_j___libc_free_0(*v29, v29[2] + 1LL);
            }
            while ( (_QWORD *)v28 != v29 );
            v28 = *(_QWORD *)(v21 + 16);
          }
          if ( v28 != v21 + 32 )
            _libc_free(v28);
        }
        while ( v21 != v73 );
        v72 = *(_QWORD *)a1;
      }
      if ( v72 != a1 + 16 )
        _libc_free(v72);
      *(_QWORD *)a1 = *(_QWORD *)a2;
      *(_DWORD *)(a1 + 8) = *(_DWORD *)(a2 + 8);
      *(_DWORD *)(a1 + 12) = *(_DWORD *)(a2 + 12);
      *(_QWORD *)(a2 + 8) = 0;
      *(_QWORD *)a2 = v70;
    }
  }
}
