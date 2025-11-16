// Function: sub_311AFF0
// Address: 0x311aff0
//
_QWORD *__fastcall sub_311AFF0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // r12
  _QWORD *i; // rdx
  _QWORD *v10; // r13
  _QWORD *v11; // rax
  unsigned __int64 v12; // rax
  int v13; // ecx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r10d
  unsigned __int64 *v17; // r8
  unsigned int v18; // edx
  unsigned __int64 *v19; // r15
  unsigned __int64 v20; // rsi
  _QWORD *v21; // r10
  unsigned int v22; // r11d
  __int64 v23; // r15
  unsigned __int64 v24; // r14
  unsigned __int64 v25; // r12
  unsigned __int64 v26; // r13
  _QWORD *v27; // r14
  __int64 v28; // r9
  _QWORD *v29; // rax
  _QWORD *v30; // rcx
  __int64 v31; // rcx
  __int64 v32; // r15
  __int64 v33; // r12
  unsigned __int64 v34; // r13
  unsigned __int64 v35; // r14
  unsigned __int64 v36; // rdi
  int v37; // eax
  _QWORD *v38; // rdx
  __int64 v39; // rdx
  _QWORD *j; // rdx
  unsigned int v41; // [rsp+4h] [rbp-7Ch]
  int v42; // [rsp+8h] [rbp-78h]
  _QWORD *v43; // [rsp+10h] [rbp-70h]
  __int64 v44; // [rsp+20h] [rbp-60h]
  __int64 v45; // [rsp+28h] [rbp-58h]
  _QWORD *v46; // [rsp+30h] [rbp-50h]
  _QWORD *v47; // [rsp+30h] [rbp-50h]
  _QWORD *v48; // [rsp+30h] [rbp-50h]
  _QWORD *v49; // [rsp+38h] [rbp-48h]
  unsigned __int64 v50[7]; // [rsp+48h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v45 = v5;
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
  result = (_QWORD *)sub_C7D670(72LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v44 = 72 * v4;
    v8 = 72 * v4 + v5;
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -1;
    }
    v10 = (_QWORD *)(v5 + 24);
    if ( v8 != v5 )
    {
      v49 = (_QWORD *)v8;
      while ( 1 )
      {
        v12 = *(v10 - 3);
        if ( v12 > 0xFFFFFFFFFFFFFFFDLL )
          goto LABEL_10;
        v13 = *(_DWORD *)(a1 + 24);
        if ( !v13 )
        {
          MEMORY[0] = 0;
          BUG();
        }
        v14 = v13 - 1;
        v15 = *(_QWORD *)(a1 + 8);
        v16 = 1;
        v17 = 0;
        v18 = v14 & (((0xBF58476D1CE4E5B9LL * v12) >> 31) ^ (484763065 * v12));
        v19 = (unsigned __int64 *)(v15 + 72LL * v18);
        v20 = *v19;
        if ( v12 != *v19 )
        {
          while ( v20 != -1 )
          {
            if ( !v17 && v20 == -2 )
              v17 = v19;
            v18 = v14 & (v16 + v18);
            v19 = (unsigned __int64 *)(v15 + 72LL * v18);
            v20 = *v19;
            if ( v12 == *v19 )
              goto LABEL_15;
            ++v16;
          }
          if ( v17 )
            v19 = v17;
        }
LABEL_15:
        *v19 = v12;
        v21 = v19 + 3;
        v19[2] = 0x600000000LL;
        v19[1] = (unsigned __int64)(v19 + 3);
        v22 = *((_DWORD *)v10 - 2);
        if ( v19 + 1 != v10 - 2 && v22 )
        {
          v27 = (_QWORD *)*(v10 - 2);
          if ( v10 == v27 )
          {
            v28 = v22;
            v29 = v10;
            if ( v22 > 6 )
            {
              v41 = *((_DWORD *)v10 - 2);
              v48 = (_QWORD *)sub_C8D7D0((__int64)(v19 + 1), (__int64)(v19 + 3), v22, 8u, v50, v22);
              sub_311AF30((__int64)(v19 + 1), v48);
              v36 = v19[1];
              v37 = v50[0];
              v38 = v48;
              v22 = v41;
              if ( v19 + 3 != (unsigned __int64 *)v36 )
              {
                v42 = v50[0];
                _libc_free(v36);
                v37 = v42;
                v38 = v48;
                v22 = v41;
              }
              v19[1] = (unsigned __int64)v38;
              v21 = v38;
              *((_DWORD *)v19 + 5) = v37;
              v29 = (_QWORD *)*(v27 - 2);
              v28 = *((unsigned int *)v27 - 2);
            }
            v30 = &v21[v28];
            if ( 8 * v28 )
            {
              do
              {
                if ( v21 )
                {
                  *v21 = *v29;
                  *v29 = 0;
                }
                ++v21;
                ++v29;
              }
              while ( v21 != v30 );
            }
            *((_DWORD *)v19 + 4) = v22;
            v31 = *(v27 - 2);
            v32 = v31 + 8LL * *((unsigned int *)v27 - 2);
            if ( v31 != v32 )
            {
              v47 = v10;
              v33 = *(v27 - 2);
              v43 = v27;
              do
              {
                v34 = *(_QWORD *)(v32 - 8);
                v32 -= 8;
                if ( v34 )
                {
                  v35 = *(_QWORD *)(v34 + 24);
                  if ( v35 )
                  {
                    sub_C7D6A0(*(_QWORD *)(v35 + 8), 16LL * *(unsigned int *)(v35 + 24), 8);
                    j_j___libc_free_0(v35);
                  }
                  j_j___libc_free_0(v34);
                }
              }
              while ( v33 != v32 );
              v10 = v47;
              v27 = v43;
            }
            *((_DWORD *)v27 - 2) = 0;
          }
          else
          {
            v19[1] = (unsigned __int64)v27;
            *((_DWORD *)v19 + 4) = *((_DWORD *)v10 - 2);
            *((_DWORD *)v19 + 5) = *((_DWORD *)v10 - 1);
            *(v10 - 2) = v10;
            *((_DWORD *)v10 - 1) = 0;
            *((_DWORD *)v10 - 2) = 0;
          }
        }
        ++*(_DWORD *)(a1 + 16);
        v23 = *(v10 - 2);
        v24 = v23 + 8LL * *((unsigned int *)v10 - 2);
        if ( v23 != v24 )
        {
          v46 = v10;
          do
          {
            v25 = *(_QWORD *)(v24 - 8);
            v24 -= 8LL;
            if ( v25 )
            {
              v26 = *(_QWORD *)(v25 + 24);
              if ( v26 )
              {
                sub_C7D6A0(*(_QWORD *)(v26 + 8), 16LL * *(unsigned int *)(v26 + 24), 8);
                j_j___libc_free_0(v26);
              }
              j_j___libc_free_0(v25);
            }
          }
          while ( v23 != v24 );
          v10 = v46;
          v24 = *(v46 - 2);
        }
        if ( v10 == (_QWORD *)v24 )
        {
LABEL_10:
          v11 = v10 + 9;
          if ( v49 == v10 + 6 )
            return (_QWORD *)sub_C7D6A0(v45, v44, 8);
        }
        else
        {
          _libc_free(v24);
          v11 = v10 + 9;
          if ( v49 == v10 + 6 )
            return (_QWORD *)sub_C7D6A0(v45, v44, 8);
        }
        v10 = v11;
      }
    }
    return (_QWORD *)sub_C7D6A0(v45, v44, 8);
  }
  else
  {
    v39 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[9 * v39]; j != result; result += 9 )
    {
      if ( result )
        *result = -1;
    }
  }
  return result;
}
