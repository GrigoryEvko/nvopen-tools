// Function: sub_13FACC0
// Address: 0x13facc0
//
__int64 __fastcall sub_13FACC0(__int64 a1)
{
  __int64 *v1; // r13
  __int64 v2; // rbx
  __int64 *v3; // r12
  __int64 v4; // r14
  __int64 *v5; // rbx
  __int64 *v6; // r15
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rax
  void *v10; // rdi
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  void *v19; // rdi
  unsigned int v20; // eax
  __int64 v21; // rdx
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdi
  __int64 v26; // rax
  __int64 v27; // rax
  void *v28; // rdi
  unsigned int v29; // eax
  __int64 v30; // rdx
  unsigned __int64 v31; // rdi
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // rax
  void *v37; // rdi
  unsigned int v38; // eax
  __int64 v39; // rdx
  unsigned __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rax
  void *v46; // rdi
  unsigned int v47; // eax
  __int64 v48; // rdx
  unsigned __int64 v49; // rdi
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 result; // rax
  __int64 v53; // rdi
  __int64 *v54; // [rsp+0h] [rbp-80h]
  __int64 *v55; // [rsp+8h] [rbp-78h]
  __int64 *v56; // [rsp+10h] [rbp-70h]
  __int64 *v57; // [rsp+18h] [rbp-68h]
  __int64 v59; // [rsp+28h] [rbp-58h]
  __int64 v60; // [rsp+30h] [rbp-50h]
  __int64 *v61; // [rsp+38h] [rbp-48h]
  __int64 v62; // [rsp+40h] [rbp-40h]
  __int64 *v63; // [rsp+48h] [rbp-38h]

  v55 = *(__int64 **)(a1 + 16);
  if ( *(__int64 **)(a1 + 8) == v55 )
  {
    *(_BYTE *)(a1 + 160) = 1;
  }
  else
  {
    v57 = *(__int64 **)(a1 + 8);
    do
    {
      v60 = *v57;
      v54 = *(__int64 **)(*v57 + 16);
      if ( *(__int64 **)(*v57 + 8) == v54 )
      {
        *(_BYTE *)(v60 + 160) = 1;
      }
      else
      {
        v56 = *(__int64 **)(*v57 + 8);
        do
        {
          v59 = *v56;
          v1 = *(__int64 **)(*v56 + 8);
          v61 = *(__int64 **)(*v56 + 16);
          if ( v1 == v61 )
          {
            *(_BYTE *)(v59 + 160) = 1;
          }
          else
          {
            do
            {
              v2 = *v1;
              v3 = *(__int64 **)(*v1 + 8);
              v63 = *(__int64 **)(*v1 + 16);
              if ( v3 == v63 )
              {
                *(_BYTE *)(v2 + 160) = 1;
              }
              else
              {
                v62 = *v1;
                do
                {
                  v4 = *v3;
                  v5 = *(__int64 **)(*v3 + 16);
                  if ( *(__int64 **)(*v3 + 8) == v5 )
                  {
                    *(_BYTE *)(v4 + 160) = 1;
                  }
                  else
                  {
                    v6 = *(__int64 **)(*v3 + 8);
                    do
                    {
                      v7 = *v6++;
                      sub_13FACC0(v7);
                    }
                    while ( v5 != v6 );
                    *(_BYTE *)(v4 + 160) = 1;
                    v8 = *(_QWORD *)(v4 + 8);
                    if ( v8 != *(_QWORD *)(v4 + 16) )
                      *(_QWORD *)(v4 + 16) = v8;
                  }
                  v9 = *(_QWORD *)(v4 + 32);
                  if ( v9 != *(_QWORD *)(v4 + 40) )
                    *(_QWORD *)(v4 + 40) = v9;
                  ++*(_QWORD *)(v4 + 56);
                  v10 = *(void **)(v4 + 72);
                  if ( v10 == *(void **)(v4 + 64) )
                  {
                    *(_QWORD *)v4 = 0;
                  }
                  else
                  {
                    v11 = 4 * (*(_DWORD *)(v4 + 84) - *(_DWORD *)(v4 + 88));
                    v12 = *(unsigned int *)(v4 + 80);
                    if ( v11 < 0x20 )
                      v11 = 32;
                    if ( (unsigned int)v12 > v11 )
                      sub_16CC920(v4 + 56);
                    else
                      memset(v10, -1, 8 * v12);
                    v13 = *(_QWORD *)(v4 + 72);
                    v14 = *(_QWORD *)(v4 + 64);
                    *(_QWORD *)v4 = 0;
                    if ( v13 != v14 )
                      _libc_free(v13);
                  }
                  v15 = *(_QWORD *)(v4 + 32);
                  if ( v15 )
                    j_j___libc_free_0(v15, *(_QWORD *)(v4 + 48) - v15);
                  v16 = *(_QWORD *)(v4 + 8);
                  if ( v16 )
                    j_j___libc_free_0(v16, *(_QWORD *)(v4 + 24) - v16);
                  ++v3;
                }
                while ( v63 != v3 );
                v2 = v62;
                *(_BYTE *)(v62 + 160) = 1;
                v17 = *(_QWORD *)(v62 + 8);
                if ( v17 != *(_QWORD *)(v62 + 16) )
                  *(_QWORD *)(v62 + 16) = v17;
              }
              v18 = *(_QWORD *)(v2 + 32);
              if ( v18 != *(_QWORD *)(v2 + 40) )
                *(_QWORD *)(v2 + 40) = v18;
              ++*(_QWORD *)(v2 + 56);
              v19 = *(void **)(v2 + 72);
              if ( v19 == *(void **)(v2 + 64) )
              {
                *(_QWORD *)v2 = 0;
              }
              else
              {
                v20 = 4 * (*(_DWORD *)(v2 + 84) - *(_DWORD *)(v2 + 88));
                v21 = *(unsigned int *)(v2 + 80);
                if ( v20 < 0x20 )
                  v20 = 32;
                if ( (unsigned int)v21 > v20 )
                  sub_16CC920(v2 + 56);
                else
                  memset(v19, -1, 8 * v21);
                v22 = *(_QWORD *)(v2 + 72);
                v23 = *(_QWORD *)(v2 + 64);
                *(_QWORD *)v2 = 0;
                if ( v22 != v23 )
                  _libc_free(v22);
              }
              v24 = *(_QWORD *)(v2 + 32);
              if ( v24 )
                j_j___libc_free_0(v24, *(_QWORD *)(v2 + 48) - v24);
              v25 = *(_QWORD *)(v2 + 8);
              if ( v25 )
                j_j___libc_free_0(v25, *(_QWORD *)(v2 + 24) - v25);
              ++v1;
            }
            while ( v61 != v1 );
            *(_BYTE *)(v59 + 160) = 1;
            v26 = *(_QWORD *)(v59 + 8);
            if ( *(_QWORD *)(v59 + 16) != v26 )
              *(_QWORD *)(v59 + 16) = v26;
          }
          v27 = *(_QWORD *)(v59 + 32);
          if ( v27 != *(_QWORD *)(v59 + 40) )
            *(_QWORD *)(v59 + 40) = v27;
          ++*(_QWORD *)(v59 + 56);
          v28 = *(void **)(v59 + 72);
          if ( v28 == *(void **)(v59 + 64) )
          {
            *(_QWORD *)v59 = 0;
          }
          else
          {
            v29 = 4 * (*(_DWORD *)(v59 + 84) - *(_DWORD *)(v59 + 88));
            v30 = *(unsigned int *)(v59 + 80);
            if ( v29 < 0x20 )
              v29 = 32;
            if ( (unsigned int)v30 > v29 )
              sub_16CC920(v59 + 56);
            else
              memset(v28, -1, 8 * v30);
            v31 = *(_QWORD *)(v59 + 72);
            v32 = *(_QWORD *)(v59 + 64);
            *(_QWORD *)v59 = 0;
            if ( v32 != v31 )
              _libc_free(v31);
          }
          v33 = *(_QWORD *)(v59 + 32);
          if ( v33 )
            j_j___libc_free_0(v33, *(_QWORD *)(v59 + 48) - v33);
          v34 = *(_QWORD *)(v59 + 8);
          if ( v34 )
            j_j___libc_free_0(v34, *(_QWORD *)(v59 + 24) - v34);
          ++v56;
        }
        while ( v54 != v56 );
        *(_BYTE *)(v60 + 160) = 1;
        v35 = *(_QWORD *)(v60 + 8);
        if ( v35 != *(_QWORD *)(v60 + 16) )
          *(_QWORD *)(v60 + 16) = v35;
      }
      v36 = *(_QWORD *)(v60 + 32);
      if ( v36 != *(_QWORD *)(v60 + 40) )
        *(_QWORD *)(v60 + 40) = v36;
      ++*(_QWORD *)(v60 + 56);
      v37 = *(void **)(v60 + 72);
      if ( v37 == *(void **)(v60 + 64) )
      {
        *(_QWORD *)v60 = 0;
      }
      else
      {
        v38 = 4 * (*(_DWORD *)(v60 + 84) - *(_DWORD *)(v60 + 88));
        v39 = *(unsigned int *)(v60 + 80);
        if ( v38 < 0x20 )
          v38 = 32;
        if ( (unsigned int)v39 > v38 )
          sub_16CC920(v60 + 56);
        else
          memset(v37, -1, 8 * v39);
        v40 = *(_QWORD *)(v60 + 72);
        v41 = *(_QWORD *)(v60 + 64);
        *(_QWORD *)v60 = 0;
        if ( v40 != v41 )
          _libc_free(v40);
      }
      v42 = *(_QWORD *)(v60 + 32);
      if ( v42 )
        j_j___libc_free_0(v42, *(_QWORD *)(v60 + 48) - v42);
      v43 = *(_QWORD *)(v60 + 8);
      if ( v43 )
        j_j___libc_free_0(v43, *(_QWORD *)(v60 + 24) - v43);
      ++v57;
    }
    while ( v55 != v57 );
    *(_BYTE *)(a1 + 160) = 1;
    v44 = *(_QWORD *)(a1 + 8);
    if ( v44 != *(_QWORD *)(a1 + 16) )
      *(_QWORD *)(a1 + 16) = v44;
  }
  v45 = *(_QWORD *)(a1 + 32);
  if ( v45 != *(_QWORD *)(a1 + 40) )
    *(_QWORD *)(a1 + 40) = v45;
  ++*(_QWORD *)(a1 + 56);
  v46 = *(void **)(a1 + 72);
  if ( v46 == *(void **)(a1 + 64) )
  {
    *(_QWORD *)a1 = 0;
  }
  else
  {
    v47 = 4 * (*(_DWORD *)(a1 + 84) - *(_DWORD *)(a1 + 88));
    v48 = *(unsigned int *)(a1 + 80);
    if ( v47 < 0x20 )
      v47 = 32;
    if ( (unsigned int)v48 > v47 )
      sub_16CC920(a1 + 56);
    else
      memset(v46, -1, 8 * v48);
    v49 = *(_QWORD *)(a1 + 72);
    v50 = *(_QWORD *)(a1 + 64);
    *(_QWORD *)a1 = 0;
    if ( v49 != v50 )
      _libc_free(v49);
  }
  v51 = *(_QWORD *)(a1 + 32);
  if ( v51 )
    j_j___libc_free_0(v51, *(_QWORD *)(a1 + 48) - v51);
  result = a1;
  v53 = *(_QWORD *)(a1 + 8);
  if ( v53 )
    return j_j___libc_free_0(v53, *(_QWORD *)(a1 + 24) - v53);
  return result;
}
