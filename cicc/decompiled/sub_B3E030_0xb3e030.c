// Function: sub_B3E030
// Address: 0xb3e030
//
void __fastcall sub_B3E030(__int64 *a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 i; // r12
  int v6; // eax
  __int64 v7; // r14
  __int64 v8; // rdx
  __int64 v9; // r13
  __int64 v10; // rbx
  __int64 v11; // rax
  _QWORD *v12; // r12
  _QWORD *v13; // r15
  _QWORD *v14; // r12
  _QWORD *v15; // rbx
  __int64 v16; // r14
  __int64 v17; // rbx
  __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // r15
  _QWORD *v21; // r12
  _QWORD *v22; // rbx
  __int64 v23; // r14
  __int64 v24; // r12
  __int64 v25; // rbx
  __int64 v26; // r15
  _QWORD *v27; // r14
  _QWORD *v28; // r15
  _QWORD *v29; // r12
  _QWORD *v30; // rbx
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // rbx
  __int64 v35; // r15
  _QWORD *v36; // r14
  _QWORD *v37; // r15
  _QWORD *v38; // r12
  _QWORD *v39; // rbx
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 v42; // r14
  unsigned __int64 v43; // r12
  __int64 v44; // rdi
  __int64 v45; // r12
  __int64 v46; // rbx
  __int64 v47; // rax
  _QWORD *v48; // r15
  _QWORD *v49; // r14
  _QWORD *v50; // r12
  _QWORD *v51; // rbx
  int v52; // r12d
  __int64 v53; // r14
  __int64 v54; // r15
  __int64 v55; // rdi
  unsigned int v56; // [rsp+4h] [rbp-6Ch]
  __int64 v57; // [rsp+8h] [rbp-68h]
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v60; // [rsp+20h] [rbp-50h]
  __int64 v61; // [rsp+28h] [rbp-48h]
  unsigned __int64 v62; // [rsp+28h] [rbp-48h]
  unsigned __int64 v63; // [rsp+28h] [rbp-48h]
  __int64 v64; // [rsp+28h] [rbp-48h]
  unsigned __int64 v65; // [rsp+28h] [rbp-48h]
  __int64 v66[7]; // [rsp+38h] [rbp-38h] BYREF

  v60 = a2;
  if ( a1 != (__int64 *)a2 )
  {
    v2 = *((unsigned int *)a1 + 2);
    v3 = *a1;
    v58 = a2 + 16;
    if ( *(_QWORD *)a2 == a2 + 16 )
    {
      v56 = *(_DWORD *)(a2 + 8);
      v57 = v56;
      if ( v56 > v2 )
      {
        if ( v56 > (unsigned __int64)*((unsigned int *)a1 + 3) )
        {
          v65 = v3 + 192 * v2;
          while ( v65 != v3 )
          {
            v65 -= 192LL;
            v45 = *(_QWORD *)(v65 + 64);
            v46 = v45 + 56LL * *(unsigned int *)(v65 + 72);
            if ( v45 != v46 )
            {
              do
              {
                v47 = *(unsigned int *)(v46 - 40);
                v48 = *(_QWORD **)(v46 - 48);
                v46 -= 56;
                v47 *= 32;
                v49 = (_QWORD *)((char *)v48 + v47);
                if ( v48 != (_QWORD *)((char *)v48 + v47) )
                {
                  do
                  {
                    v49 -= 4;
                    if ( (_QWORD *)*v49 != v49 + 2 )
                    {
                      a2 = v49[2] + 1LL;
                      j_j___libc_free_0(*v49, a2);
                    }
                  }
                  while ( v48 != v49 );
                  v48 = *(_QWORD **)(v46 + 8);
                }
                if ( v48 != (_QWORD *)(v46 + 24) )
                  _libc_free(v48, a2);
              }
              while ( v45 != v46 );
              v45 = *(_QWORD *)(v65 + 64);
            }
            if ( v45 != v65 + 80 )
              _libc_free(v45, a2);
            v50 = *(_QWORD **)(v65 + 16);
            v51 = &v50[4 * *(unsigned int *)(v65 + 24)];
            if ( v50 != v51 )
            {
              do
              {
                v51 -= 4;
                if ( (_QWORD *)*v51 != v51 + 2 )
                {
                  a2 = v51[2] + 1LL;
                  j_j___libc_free_0(*v51, a2);
                }
              }
              while ( v50 != v51 );
              v50 = *(_QWORD **)(v65 + 16);
            }
            if ( v50 != (_QWORD *)(v65 + 32) )
              _libc_free(v50, a2);
          }
          *((_DWORD *)a1 + 2) = 0;
          a2 = sub_C8D7D0(a1, a1 + 2, v56, 192, v66);
          v3 = a2;
          sub_B3DE10(a1, a2);
          v52 = v66[0];
          if ( a1 + 2 != (__int64 *)*a1 )
            _libc_free(*a1, a2);
          *a1 = a2;
          *((_DWORD *)a1 + 3) = v52;
          v4 = *(_QWORD *)v60;
          v58 = *(_QWORD *)v60;
          v57 = *(unsigned int *)(v60 + 8);
        }
        else
        {
          v4 = a2 + 16;
          if ( *((_DWORD *)a1 + 2) )
          {
            v40 = v3 + 16;
            v41 = 192 * v2;
            v42 = a2 + 32;
            v43 = v40 + 192 * v2;
            do
            {
              *(_DWORD *)(v40 - 16) = *(_DWORD *)(v42 - 16);
              *(_DWORD *)(v40 - 12) = *(_DWORD *)(v42 - 12);
              *(_BYTE *)(v40 - 8) = *(_BYTE *)(v42 - 8);
              *(_BYTE *)(v40 - 7) = *(_BYTE *)(v42 - 7);
              *(_BYTE *)(v40 - 6) = *(_BYTE *)(v42 - 6);
              *(_BYTE *)(v40 - 5) = *(_BYTE *)(v42 - 5);
              *(_DWORD *)(v40 - 4) = *(_DWORD *)(v42 - 4);
              sub_B3BE00(v40, v42);
              a2 = v42 + 48;
              v44 = v40 + 48;
              v40 += 192;
              sub_B3D620(v44, v42 + 48);
              v42 += 192;
            }
            while ( v40 != v43 );
            v3 = v41 + *a1;
            v58 = *(_QWORD *)v60;
            v4 = *(_QWORD *)v60 + v41;
            v57 = *(unsigned int *)(v60 + 8);
          }
        }
        for ( i = v58 + 192 * v57; i != v4; v3 += 192 )
        {
          if ( v3 )
          {
            *(_DWORD *)v3 = *(_DWORD *)v4;
            *(_DWORD *)(v3 + 4) = *(_DWORD *)(v4 + 4);
            *(_BYTE *)(v3 + 8) = *(_BYTE *)(v4 + 8);
            *(_BYTE *)(v3 + 9) = *(_BYTE *)(v4 + 9);
            *(_BYTE *)(v3 + 10) = *(_BYTE *)(v4 + 10);
            *(_BYTE *)(v3 + 11) = *(_BYTE *)(v4 + 11);
            v6 = *(_DWORD *)(v4 + 12);
            *(_DWORD *)(v3 + 24) = 0;
            *(_DWORD *)(v3 + 12) = v6;
            *(_QWORD *)(v3 + 16) = v3 + 32;
            *(_DWORD *)(v3 + 28) = 1;
            if ( *(_DWORD *)(v4 + 24) )
            {
              a2 = v4 + 16;
              sub_B3BE00(v3 + 16, v4 + 16);
            }
            *(_DWORD *)(v3 + 72) = 0;
            *(_QWORD *)(v3 + 64) = v3 + 80;
            *(_DWORD *)(v3 + 76) = 2;
            if ( *(_DWORD *)(v4 + 72) )
            {
              a2 = v4 + 64;
              sub_B3D620(v3 + 64, v4 + 64);
            }
          }
          v4 += 192;
        }
        *((_DWORD *)a1 + 2) = v56;
        v61 = *(_QWORD *)v60;
        v7 = *(_QWORD *)v60 + 192LL * *(unsigned int *)(v60 + 8);
        if ( *(_QWORD *)v60 != v7 )
        {
          do
          {
            v8 = *(unsigned int *)(v7 - 120);
            v9 = *(_QWORD *)(v7 - 128);
            v7 -= 192;
            v10 = v9 + 56 * v8;
            if ( v9 != v10 )
            {
              do
              {
                v11 = *(unsigned int *)(v10 - 40);
                v12 = *(_QWORD **)(v10 - 48);
                v10 -= 56;
                v11 *= 32;
                v13 = (_QWORD *)((char *)v12 + v11);
                if ( v12 != (_QWORD *)((char *)v12 + v11) )
                {
                  do
                  {
                    v13 -= 4;
                    if ( (_QWORD *)*v13 != v13 + 2 )
                    {
                      a2 = v13[2] + 1LL;
                      j_j___libc_free_0(*v13, a2);
                    }
                  }
                  while ( v12 != v13 );
                  v12 = *(_QWORD **)(v10 + 8);
                }
                if ( v12 != (_QWORD *)(v10 + 24) )
                  _libc_free(v12, a2);
              }
              while ( v9 != v10 );
              v9 = *(_QWORD *)(v7 + 64);
            }
            if ( v9 != v7 + 80 )
              _libc_free(v9, a2);
            v14 = *(_QWORD **)(v7 + 16);
            v15 = &v14[4 * *(unsigned int *)(v7 + 24)];
            if ( v14 != v15 )
            {
              do
              {
                v15 -= 4;
                if ( (_QWORD *)*v15 != v15 + 2 )
                {
                  a2 = v15[2] + 1LL;
                  j_j___libc_free_0(*v15, a2);
                }
              }
              while ( v14 != v15 );
              v14 = *(_QWORD **)(v7 + 16);
            }
            if ( v14 != (_QWORD *)(v7 + 32) )
              _libc_free(v14, a2);
          }
          while ( v61 != v7 );
        }
LABEL_35:
        *(_DWORD *)(v60 + 8) = 0;
        return;
      }
      v23 = *a1;
      if ( v56 )
      {
        v53 = v3 + 16;
        v54 = a2 + 32;
        do
        {
          *(_DWORD *)(v53 - 16) = *(_DWORD *)(v54 - 16);
          *(_DWORD *)(v53 - 12) = *(_DWORD *)(v54 - 12);
          *(_BYTE *)(v53 - 8) = *(_BYTE *)(v54 - 8);
          *(_BYTE *)(v53 - 7) = *(_BYTE *)(v54 - 7);
          *(_BYTE *)(v53 - 6) = *(_BYTE *)(v54 - 6);
          *(_BYTE *)(v53 - 5) = *(_BYTE *)(v54 - 5);
          *(_DWORD *)(v53 - 4) = *(_DWORD *)(v54 - 4);
          sub_B3BE00(v53, v54);
          a2 = v54 + 48;
          v55 = v53 + 48;
          v53 += 192;
          sub_B3D620(v55, v54 + 48);
          v54 += 192;
        }
        while ( v3 + 16 + 192LL * v56 != v53 );
        v3 += 192LL * v56;
        v23 = *a1;
        v2 = *((unsigned int *)a1 + 2);
      }
      v63 = v23 + 192 * v2;
      while ( v3 != v63 )
      {
        v63 -= 192LL;
        v24 = *(_QWORD *)(v63 + 64);
        v25 = v24 + 56LL * *(unsigned int *)(v63 + 72);
        if ( v24 != v25 )
        {
          do
          {
            v26 = *(unsigned int *)(v25 - 40);
            v27 = *(_QWORD **)(v25 - 48);
            v25 -= 56;
            v28 = &v27[4 * v26];
            if ( v27 != v28 )
            {
              do
              {
                v28 -= 4;
                if ( (_QWORD *)*v28 != v28 + 2 )
                {
                  a2 = v28[2] + 1LL;
                  j_j___libc_free_0(*v28, a2);
                }
              }
              while ( v27 != v28 );
              v27 = *(_QWORD **)(v25 + 8);
            }
            if ( v27 != (_QWORD *)(v25 + 24) )
              _libc_free(v27, a2);
          }
          while ( v24 != v25 );
          v24 = *(_QWORD *)(v63 + 64);
        }
        if ( v24 != v63 + 80 )
          _libc_free(v24, a2);
        v29 = *(_QWORD **)(v63 + 16);
        v30 = &v29[4 * *(unsigned int *)(v63 + 24)];
        if ( v29 != v30 )
        {
          do
          {
            v30 -= 4;
            if ( (_QWORD *)*v30 != v30 + 2 )
            {
              a2 = v30[2] + 1LL;
              j_j___libc_free_0(*v30, a2);
            }
          }
          while ( v29 != v30 );
          v29 = *(_QWORD **)(v63 + 16);
        }
        if ( v29 != (_QWORD *)(v63 + 32) )
          _libc_free(v29, a2);
      }
      *((_DWORD *)a1 + 2) = v56;
      v64 = *(_QWORD *)v60;
      v31 = *(_QWORD *)v60 + 192LL * *(unsigned int *)(v60 + 8);
      if ( *(_QWORD *)v60 == v31 )
        goto LABEL_35;
      do
      {
        v32 = *(unsigned int *)(v31 - 120);
        v33 = *(_QWORD *)(v31 - 128);
        v31 -= 192;
        v34 = v33 + 56 * v32;
        if ( v33 != v34 )
        {
          do
          {
            v35 = *(unsigned int *)(v34 - 40);
            v36 = *(_QWORD **)(v34 - 48);
            v34 -= 56;
            v37 = &v36[4 * v35];
            if ( v36 != v37 )
            {
              do
              {
                v37 -= 4;
                if ( (_QWORD *)*v37 != v37 + 2 )
                {
                  a2 = v37[2] + 1LL;
                  j_j___libc_free_0(*v37, a2);
                }
              }
              while ( v36 != v37 );
              v36 = *(_QWORD **)(v34 + 8);
            }
            if ( v36 != (_QWORD *)(v34 + 24) )
              _libc_free(v36, a2);
          }
          while ( v33 != v34 );
          v33 = *(_QWORD *)(v31 + 64);
        }
        if ( v33 != v31 + 80 )
          _libc_free(v33, a2);
        v38 = *(_QWORD **)(v31 + 16);
        v39 = &v38[4 * *(unsigned int *)(v31 + 24)];
        if ( v38 != v39 )
        {
          do
          {
            v39 -= 4;
            if ( (_QWORD *)*v39 != v39 + 2 )
            {
              a2 = v39[2] + 1LL;
              j_j___libc_free_0(*v39, a2);
            }
          }
          while ( v38 != v39 );
          v38 = *(_QWORD **)(v31 + 16);
        }
        if ( v38 != (_QWORD *)(v31 + 32) )
          _libc_free(v38, a2);
      }
      while ( v64 != v31 );
      *(_DWORD *)(v60 + 8) = 0;
    }
    else
    {
      v62 = v3 + 192 * v2;
      if ( v62 != v3 )
      {
        do
        {
          v62 -= 192LL;
          v16 = *(_QWORD *)(v62 + 64);
          v17 = v16 + 56LL * *(unsigned int *)(v62 + 72);
          if ( v16 != v17 )
          {
            do
            {
              v18 = *(unsigned int *)(v17 - 40);
              v19 = *(_QWORD **)(v17 - 48);
              v17 -= 56;
              v18 *= 32;
              v20 = (_QWORD *)((char *)v19 + v18);
              if ( v19 != (_QWORD *)((char *)v19 + v18) )
              {
                do
                {
                  v20 -= 4;
                  if ( (_QWORD *)*v20 != v20 + 2 )
                  {
                    a2 = v20[2] + 1LL;
                    j_j___libc_free_0(*v20, a2);
                  }
                }
                while ( v19 != v20 );
                v19 = *(_QWORD **)(v17 + 8);
              }
              if ( v19 != (_QWORD *)(v17 + 24) )
                _libc_free(v19, a2);
            }
            while ( v16 != v17 );
            v16 = *(_QWORD *)(v62 + 64);
          }
          if ( v16 != v62 + 80 )
            _libc_free(v16, a2);
          v21 = *(_QWORD **)(v62 + 16);
          v22 = &v21[4 * *(unsigned int *)(v62 + 24)];
          if ( v21 != v22 )
          {
            do
            {
              v22 -= 4;
              if ( (_QWORD *)*v22 != v22 + 2 )
              {
                a2 = v22[2] + 1LL;
                j_j___libc_free_0(*v22, a2);
              }
            }
            while ( v21 != v22 );
            v21 = *(_QWORD **)(v62 + 16);
          }
          if ( v21 != (_QWORD *)(v62 + 32) )
            _libc_free(v21, a2);
        }
        while ( v62 != v3 );
        v3 = *a1;
      }
      if ( (__int64 *)v3 != a1 + 2 )
        _libc_free(v3, a2);
      *a1 = *(_QWORD *)v60;
      *((_DWORD *)a1 + 2) = *(_DWORD *)(v60 + 8);
      *((_DWORD *)a1 + 3) = *(_DWORD *)(v60 + 12);
      *(_QWORD *)(v60 + 8) = 0;
      *(_QWORD *)v60 = v58;
    }
  }
}
