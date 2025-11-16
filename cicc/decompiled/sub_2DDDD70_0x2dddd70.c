// Function: sub_2DDDD70
// Address: 0x2dddd70
//
_QWORD *__fastcall sub_2DDDD70(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // r12
  __int64 v4; // rbx
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  _QWORD *v9; // rbx
  _QWORD *v10; // rax
  unsigned __int64 v11; // rax
  int v12; // ecx
  int v13; // ecx
  __int64 v14; // rdi
  int v15; // r10d
  unsigned int v16; // edx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rsi
  _QWORD *v20; // r15
  unsigned int v21; // r10d
  __int64 v22; // r14
  unsigned __int64 v23; // r15
  unsigned __int64 v24; // r13
  unsigned __int64 v25; // r13
  unsigned __int64 v26; // rdi
  _QWORD *v27; // r14
  _QWORD *v28; // rax
  __int64 v29; // rcx
  bool v30; // zf
  __int64 v31; // rcx
  _QWORD *v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 v37; // r13
  unsigned __int64 v38; // r12
  unsigned __int64 v39; // r12
  unsigned __int64 v40; // rdi
  int v41; // eax
  _QWORD *v42; // rdx
  unsigned __int64 v43; // rdi
  _QWORD *j; // rdx
  __int64 v45; // [rsp+0h] [rbp-80h]
  unsigned int v46; // [rsp+8h] [rbp-78h]
  int v47; // [rsp+8h] [rbp-78h]
  __int64 v48; // [rsp+10h] [rbp-70h]
  _QWORD *v49; // [rsp+10h] [rbp-70h]
  __int64 v50; // [rsp+18h] [rbp-68h]
  _QWORD *v51; // [rsp+18h] [rbp-68h]
  unsigned int v52; // [rsp+18h] [rbp-68h]
  __int64 v53; // [rsp+28h] [rbp-58h]
  __int64 v54; // [rsp+30h] [rbp-50h]
  _QWORD *v55; // [rsp+38h] [rbp-48h]
  unsigned __int64 v56[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v3 = a1;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v54 = v5;
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v53 = 56 * v4;
    v55 = (_QWORD *)(56 * v4 + v5);
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -1;
    }
    v9 = (_QWORD *)(v5 + 24);
    if ( v55 != (_QWORD *)v5 )
    {
      while ( 1 )
      {
        v11 = *(v9 - 3);
        if ( v11 > 0xFFFFFFFFFFFFFFFDLL )
          goto LABEL_10;
        v12 = *(_DWORD *)(v3 + 24);
        if ( !v12 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v13 = v12 - 1;
        v14 = *(_QWORD *)(v3 + 8);
        v15 = 1;
        v16 = v13 & (((0xBF58476D1CE4E5B9LL * v11) >> 31) ^ (484763065 * v11));
        v17 = 0;
        v18 = v14 + 56LL * v16;
        v19 = *(_QWORD *)v18;
        if ( v11 != *(_QWORD *)v18 )
        {
          while ( v19 != -1 )
          {
            if ( !v17 && v19 == -2 )
              v17 = v18;
            v16 = v13 & (v15 + v16);
            v18 = v14 + 56LL * v16;
            v19 = *(_QWORD *)v18;
            if ( v11 == *(_QWORD *)v18 )
              goto LABEL_15;
            ++v15;
          }
          if ( v17 )
            v18 = v17;
        }
LABEL_15:
        *(_QWORD *)v18 = v11;
        v20 = (_QWORD *)(v18 + 24);
        *(_QWORD *)(v18 + 16) = 0x100000000LL;
        *(_QWORD *)(v18 + 8) = v18 + 24;
        v21 = *((_DWORD *)v9 - 2);
        if ( (_QWORD *)(v18 + 8) != v9 - 2 && v21 )
        {
          v27 = (_QWORD *)*(v9 - 2);
          if ( v9 == v27 )
          {
            v28 = v9;
            v29 = 1;
            if ( v21 != 1 )
            {
              v45 = v18;
              v46 = *((_DWORD *)v9 - 2);
              v48 = v18 + 8;
              v51 = (_QWORD *)sub_C8D7D0(v18 + 8, v18 + 24, v21, 0x20u, v56, v18);
              sub_2DDD850(v48, v51);
              v18 = v45;
              v41 = v56[0];
              v42 = v51;
              v21 = v46;
              v43 = *(_QWORD *)(v45 + 8);
              if ( v20 != (_QWORD *)v43 )
              {
                v47 = v56[0];
                v49 = v51;
                v52 = v21;
                _libc_free(v43);
                v18 = v45;
                v41 = v47;
                v42 = v49;
                v21 = v52;
              }
              *(_QWORD *)(v18 + 8) = v42;
              v20 = v42;
              *(_DWORD *)(v18 + 20) = v41;
              v28 = (_QWORD *)*(v27 - 2);
              v29 = *((unsigned int *)v27 - 2);
            }
            v31 = 4 * v29;
            v30 = v31 == 0;
            v32 = &v20[v31];
            if ( !v30 )
            {
              do
              {
                if ( v20 )
                {
                  *v20 = *v28;
                  v20[1] = v28[1];
                  v20[2] = v28[2];
                  v33 = v28[3];
                  v28[2] = 0;
                  v20[3] = v33;
                  v28[3] = 0;
                }
                v20 += 4;
                v28 += 4;
              }
              while ( v20 != v32 );
            }
            *(_DWORD *)(v18 + 16) = v21;
            v34 = *(v27 - 2);
            v35 = 32LL * *((unsigned int *)v27 - 2);
            v36 = v34 + v35;
            if ( v34 != v34 + v35 )
            {
              v50 = v3;
              v37 = *(v27 - 2);
              do
              {
                v38 = *(_QWORD *)(v36 - 8);
                v36 -= 32;
                if ( v38 )
                {
                  sub_C7D6A0(*(_QWORD *)(v38 + 8), 16LL * *(unsigned int *)(v38 + 24), 8);
                  j_j___libc_free_0(v38);
                }
                v39 = *(_QWORD *)(v36 + 16);
                if ( v39 )
                {
                  v40 = *(_QWORD *)(v39 + 32);
                  if ( v40 != v39 + 48 )
                    _libc_free(v40);
                  sub_C7D6A0(*(_QWORD *)(v39 + 8), 8LL * *(unsigned int *)(v39 + 24), 4);
                  j_j___libc_free_0(v39);
                }
              }
              while ( v37 != v36 );
              v3 = v50;
            }
            *((_DWORD *)v27 - 2) = 0;
          }
          else
          {
            *(_QWORD *)(v18 + 8) = v27;
            *(_DWORD *)(v18 + 16) = *((_DWORD *)v9 - 2);
            *(_DWORD *)(v18 + 20) = *((_DWORD *)v9 - 1);
            *(v9 - 2) = v9;
            *((_DWORD *)v9 - 1) = 0;
            *((_DWORD *)v9 - 2) = 0;
          }
        }
        ++*(_DWORD *)(v3 + 16);
        v22 = *(v9 - 2);
        v23 = v22 + 32LL * *((unsigned int *)v9 - 2);
        if ( v22 != v23 )
        {
          do
          {
            v24 = *(_QWORD *)(v23 - 8);
            v23 -= 32LL;
            if ( v24 )
            {
              sub_C7D6A0(*(_QWORD *)(v24 + 8), 16LL * *(unsigned int *)(v24 + 24), 8);
              j_j___libc_free_0(v24);
            }
            v25 = *(_QWORD *)(v23 + 16);
            if ( v25 )
            {
              v26 = *(_QWORD *)(v25 + 32);
              if ( v26 != v25 + 48 )
                _libc_free(v26);
              sub_C7D6A0(*(_QWORD *)(v25 + 8), 8LL * *(unsigned int *)(v25 + 24), 4);
              j_j___libc_free_0(v25);
            }
          }
          while ( v22 != v23 );
          v23 = *(v9 - 2);
        }
        if ( v9 == (_QWORD *)v23 )
        {
LABEL_10:
          v10 = v9 + 7;
          if ( v55 == v9 + 4 )
            return (_QWORD *)sub_C7D6A0(v54, v53, 8);
        }
        else
        {
          _libc_free(v23);
          v10 = v9 + 7;
          if ( v55 == v9 + 4 )
            return (_QWORD *)sub_C7D6A0(v54, v53, 8);
        }
        v9 = v10;
      }
    }
    return (_QWORD *)sub_C7D6A0(v54, v53, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * *(unsigned int *)(a1 + 24)]; j != result; result += 7 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
