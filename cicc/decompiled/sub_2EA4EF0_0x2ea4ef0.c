// Function: sub_2EA4EF0
// Address: 0x2ea4ef0
//
void __fastcall sub_2EA4EF0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r12
  __int64 v3; // r15
  __int64 *v4; // r13
  __int64 v5; // rbx
  __int64 *v6; // r12
  __int64 *v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // rdx
  char v13; // al
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned int v18; // eax
  __int64 v19; // rdx
  char v20; // al
  unsigned __int64 v21; // rdi
  unsigned __int64 v22; // rdi
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rdx
  char v27; // al
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rdx
  char v34; // al
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rax
  unsigned int v39; // eax
  __int64 v40; // rdx
  char v41; // al
  unsigned __int64 v42; // rdi
  unsigned __int64 v43; // rdi
  __int64 *v44; // [rsp+0h] [rbp-80h]
  __int64 *v45; // [rsp+8h] [rbp-78h]
  __int64 *v46; // [rsp+10h] [rbp-70h]
  __int64 *v47; // [rsp+18h] [rbp-68h]
  __int64 v49; // [rsp+28h] [rbp-58h]
  __int64 v50; // [rsp+30h] [rbp-50h]
  __int64 *v51; // [rsp+38h] [rbp-48h]
  __int64 *v52; // [rsp+40h] [rbp-40h]
  __int64 *v53; // [rsp+48h] [rbp-38h]

  v45 = *(__int64 **)(a1 + 16);
  if ( *(__int64 **)(a1 + 8) == v45 )
  {
    *(_BYTE *)(a1 + 152) = 1;
  }
  else
  {
    v47 = *(__int64 **)(a1 + 8);
    do
    {
      v50 = *v47;
      v44 = *(__int64 **)(*v47 + 16);
      if ( *(__int64 **)(*v47 + 8) == v44 )
      {
        *(_BYTE *)(v50 + 152) = 1;
      }
      else
      {
        v46 = *(__int64 **)(*v47 + 8);
        do
        {
          v49 = *v46;
          v2 = *(__int64 **)(*v46 + 8);
          v51 = *(__int64 **)(*v46 + 16);
          if ( v2 == v51 )
          {
            *(_BYTE *)(v49 + 152) = 1;
          }
          else
          {
            do
            {
              v3 = *v2;
              v4 = *(__int64 **)(*v2 + 8);
              v53 = *(__int64 **)(*v2 + 16);
              if ( v4 == v53 )
              {
                *(_BYTE *)(v3 + 152) = 1;
              }
              else
              {
                v52 = v2;
                do
                {
                  v5 = *v4;
                  v6 = *(__int64 **)(*v4 + 16);
                  if ( *(__int64 **)(*v4 + 8) == v6 )
                  {
                    *(_BYTE *)(v5 + 152) = 1;
                  }
                  else
                  {
                    v7 = *(__int64 **)(*v4 + 8);
                    do
                    {
                      v8 = *v7++;
                      sub_2EA4EF0(v8);
                    }
                    while ( v6 != v7 );
                    *(_BYTE *)(v5 + 152) = 1;
                    v9 = *(_QWORD *)(v5 + 8);
                    if ( *(_QWORD *)(v5 + 16) != v9 )
                      *(_QWORD *)(v5 + 16) = v9;
                  }
                  v10 = *(_QWORD *)(v5 + 32);
                  if ( v10 != *(_QWORD *)(v5 + 40) )
                    *(_QWORD *)(v5 + 40) = v10;
                  ++*(_QWORD *)(v5 + 56);
                  if ( *(_BYTE *)(v5 + 84) )
                  {
                    *(_QWORD *)v5 = 0;
                  }
                  else
                  {
                    v11 = 4 * (*(_DWORD *)(v5 + 76) - *(_DWORD *)(v5 + 80));
                    v12 = *(unsigned int *)(v5 + 72);
                    if ( v11 < 0x20 )
                      v11 = 32;
                    if ( (unsigned int)v12 > v11 )
                    {
                      sub_C8C990(v5 + 56, a2);
                    }
                    else
                    {
                      a2 = 0xFFFFFFFFLL;
                      memset(*(void **)(v5 + 64), -1, 8 * v12);
                    }
                    v13 = *(_BYTE *)(v5 + 84);
                    *(_QWORD *)v5 = 0;
                    if ( !v13 )
                      _libc_free(*(_QWORD *)(v5 + 64));
                  }
                  v14 = *(_QWORD *)(v5 + 32);
                  if ( v14 )
                  {
                    a2 = *(_QWORD *)(v5 + 48) - v14;
                    j_j___libc_free_0(v14);
                  }
                  v15 = *(_QWORD *)(v5 + 8);
                  if ( v15 )
                  {
                    a2 = *(_QWORD *)(v5 + 24) - v15;
                    j_j___libc_free_0(v15);
                  }
                  ++v4;
                }
                while ( v53 != v4 );
                v2 = v52;
                v16 = *(_QWORD *)(v3 + 8);
                *(_BYTE *)(v3 + 152) = 1;
                if ( v16 != *(_QWORD *)(v3 + 16) )
                  *(_QWORD *)(v3 + 16) = v16;
              }
              v17 = *(_QWORD *)(v3 + 32);
              if ( v17 != *(_QWORD *)(v3 + 40) )
                *(_QWORD *)(v3 + 40) = v17;
              ++*(_QWORD *)(v3 + 56);
              if ( *(_BYTE *)(v3 + 84) )
              {
                *(_QWORD *)v3 = 0;
              }
              else
              {
                v18 = 4 * (*(_DWORD *)(v3 + 76) - *(_DWORD *)(v3 + 80));
                v19 = *(unsigned int *)(v3 + 72);
                if ( v18 < 0x20 )
                  v18 = 32;
                if ( (unsigned int)v19 > v18 )
                {
                  sub_C8C990(v3 + 56, a2);
                }
                else
                {
                  a2 = 0xFFFFFFFFLL;
                  memset(*(void **)(v3 + 64), -1, 8 * v19);
                }
                v20 = *(_BYTE *)(v3 + 84);
                *(_QWORD *)v3 = 0;
                if ( !v20 )
                  _libc_free(*(_QWORD *)(v3 + 64));
              }
              v21 = *(_QWORD *)(v3 + 32);
              if ( v21 )
              {
                a2 = *(_QWORD *)(v3 + 48) - v21;
                j_j___libc_free_0(v21);
              }
              v22 = *(_QWORD *)(v3 + 8);
              if ( v22 )
              {
                a2 = *(_QWORD *)(v3 + 24) - v22;
                j_j___libc_free_0(v22);
              }
              ++v2;
            }
            while ( v51 != v2 );
            *(_BYTE *)(v49 + 152) = 1;
            v23 = *(_QWORD *)(v49 + 8);
            if ( v23 != *(_QWORD *)(v49 + 16) )
              *(_QWORD *)(v49 + 16) = v23;
          }
          v24 = *(_QWORD *)(v49 + 32);
          if ( v24 != *(_QWORD *)(v49 + 40) )
            *(_QWORD *)(v49 + 40) = v24;
          ++*(_QWORD *)(v49 + 56);
          if ( *(_BYTE *)(v49 + 84) )
          {
            *(_QWORD *)v49 = 0;
          }
          else
          {
            v25 = 4 * (*(_DWORD *)(v49 + 76) - *(_DWORD *)(v49 + 80));
            v26 = *(unsigned int *)(v49 + 72);
            if ( v25 < 0x20 )
              v25 = 32;
            if ( (unsigned int)v26 > v25 )
            {
              sub_C8C990(v49 + 56, a2);
            }
            else
            {
              a2 = 0xFFFFFFFFLL;
              memset(*(void **)(v49 + 64), -1, 8 * v26);
            }
            v27 = *(_BYTE *)(v49 + 84);
            *(_QWORD *)v49 = 0;
            if ( !v27 )
              _libc_free(*(_QWORD *)(v49 + 64));
          }
          v28 = *(_QWORD *)(v49 + 32);
          if ( v28 )
          {
            a2 = *(_QWORD *)(v49 + 48) - v28;
            j_j___libc_free_0(v28);
          }
          v29 = *(_QWORD *)(v49 + 8);
          if ( v29 )
          {
            a2 = *(_QWORD *)(v49 + 24) - v29;
            j_j___libc_free_0(v29);
          }
          ++v46;
        }
        while ( v44 != v46 );
        *(_BYTE *)(v50 + 152) = 1;
        v30 = *(_QWORD *)(v50 + 8);
        if ( *(_QWORD *)(v50 + 16) != v30 )
          *(_QWORD *)(v50 + 16) = v30;
      }
      v31 = *(_QWORD *)(v50 + 32);
      if ( v31 != *(_QWORD *)(v50 + 40) )
        *(_QWORD *)(v50 + 40) = v31;
      ++*(_QWORD *)(v50 + 56);
      if ( *(_BYTE *)(v50 + 84) )
      {
        *(_QWORD *)v50 = 0;
      }
      else
      {
        v32 = 4 * (*(_DWORD *)(v50 + 76) - *(_DWORD *)(v50 + 80));
        v33 = *(unsigned int *)(v50 + 72);
        if ( v32 < 0x20 )
          v32 = 32;
        if ( (unsigned int)v33 > v32 )
        {
          sub_C8C990(v50 + 56, a2);
        }
        else
        {
          a2 = 0xFFFFFFFFLL;
          memset(*(void **)(v50 + 64), -1, 8 * v33);
        }
        v34 = *(_BYTE *)(v50 + 84);
        *(_QWORD *)v50 = 0;
        if ( !v34 )
          _libc_free(*(_QWORD *)(v50 + 64));
      }
      v35 = *(_QWORD *)(v50 + 32);
      if ( v35 )
      {
        a2 = *(_QWORD *)(v50 + 48) - v35;
        j_j___libc_free_0(v35);
      }
      v36 = *(_QWORD *)(v50 + 8);
      if ( v36 )
      {
        a2 = *(_QWORD *)(v50 + 24) - v36;
        j_j___libc_free_0(v36);
      }
      ++v47;
    }
    while ( v45 != v47 );
    *(_BYTE *)(a1 + 152) = 1;
    v37 = *(_QWORD *)(a1 + 8);
    if ( v37 != *(_QWORD *)(a1 + 16) )
      *(_QWORD *)(a1 + 16) = v37;
  }
  v38 = *(_QWORD *)(a1 + 32);
  if ( v38 != *(_QWORD *)(a1 + 40) )
    *(_QWORD *)(a1 + 40) = v38;
  ++*(_QWORD *)(a1 + 56);
  if ( *(_BYTE *)(a1 + 84) )
  {
    *(_QWORD *)a1 = 0;
  }
  else
  {
    v39 = 4 * (*(_DWORD *)(a1 + 76) - *(_DWORD *)(a1 + 80));
    v40 = *(unsigned int *)(a1 + 72);
    if ( v39 < 0x20 )
      v39 = 32;
    if ( (unsigned int)v40 > v39 )
      sub_C8C990(a1 + 56, a2);
    else
      memset(*(void **)(a1 + 64), -1, 8 * v40);
    v41 = *(_BYTE *)(a1 + 84);
    *(_QWORD *)a1 = 0;
    if ( !v41 )
      _libc_free(*(_QWORD *)(a1 + 64));
  }
  v42 = *(_QWORD *)(a1 + 32);
  if ( v42 )
    j_j___libc_free_0(v42);
  v43 = *(_QWORD *)(a1 + 8);
  if ( v43 )
    j_j___libc_free_0(v43);
}
