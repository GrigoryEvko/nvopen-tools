// Function: sub_34A34A0
// Address: 0x34a34a0
//
__int64 *__fastcall sub_34A34A0(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // r13
  __int64 v10; // rax
  bool v11; // zf
  __int64 *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 *v15; // rbx
  __int64 *v16; // rcx
  __int64 v17; // r13
  _QWORD *i; // rdx
  __int64 *v19; // rbx
  __int64 v20; // rsi
  int v21; // edi
  int v22; // r10d
  unsigned __int64 v23; // r9
  __int64 v24; // rcx
  __int64 *v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rdx
  unsigned __int64 v28; // r8
  __int64 v29; // rax
  int v30; // edx
  __int64 *v31; // rax
  __int64 *v32; // r13
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rdx
  __int64 *j; // rdx
  __int64 *result; // rax
  __int64 *v39; // r8
  int v40; // edi
  int v41; // r14d
  __int64 *v42; // r10
  unsigned int v43; // esi
  __int64 *v44; // rdx
  __int64 v45; // r9
  __int64 v46; // rcx
  int v47; // edi
  unsigned __int64 v48; // [rsp+8h] [rbp-78h]
  _BYTE v49[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    v15 = (__int64 *)(a1 + 16);
    v16 = (__int64 *)(a1 + 80);
    if ( !v5 )
    {
      v17 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      v9 = 16 * v17;
      v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
      *(_QWORD *)(a1 + 8) &= 1uLL;
      v12 = (__int64 *)(v4 + v9);
      if ( v11 )
        goto LABEL_6;
      goto LABEL_9;
    }
  }
  else
  {
    v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
            | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
            | (a2 - 1)
            | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
          | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
        | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1))
       + 1;
    v2 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v15 = (__int64 *)(a1 + 16);
      v16 = (__int64 *)(a1 + 80);
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 16LL * (unsigned int)v6;
LABEL_5:
        v9 = 16 * v7;
        v10 = sub_C7D670(v8, 8);
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_DWORD *)(a1 + 24) = v2;
        v12 = (__int64 *)(v4 + v9);
        *(_QWORD *)(a1 + 16) = v10;
        if ( v11 )
        {
LABEL_6:
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
LABEL_10:
          for ( i = &v13[v14]; i != v13; v13 += 2 )
          {
            if ( v13 )
              *v13 = -4096;
          }
          v19 = (__int64 *)v4;
          if ( (__int64 *)v4 != v12 )
          {
            do
            {
              v29 = *v19;
              if ( *v19 != -4096 && v29 != -8192 )
              {
                if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
                {
                  v20 = a1 + 16;
                  v21 = 3;
                }
                else
                {
                  v30 = *(_DWORD *)(a1 + 24);
                  v20 = *(_QWORD *)(a1 + 16);
                  if ( !v30 )
                  {
                    MEMORY[0] = *v19;
                    BUG();
                  }
                  v21 = v30 - 1;
                }
                v22 = 1;
                v23 = 0;
                v24 = v21 & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
                v25 = (__int64 *)(v20 + 16 * v24);
                v26 = *v25;
                if ( *v25 != v29 )
                {
                  while ( v26 != -4096 )
                  {
                    if ( v26 == -8192 && !v23 )
                      v23 = (unsigned __int64)v25;
                    v24 = v21 & (unsigned int)(v22 + v24);
                    v25 = (__int64 *)(v20 + 16LL * (unsigned int)v24);
                    v26 = *v25;
                    if ( v29 == *v25 )
                      goto LABEL_18;
                    ++v22;
                  }
                  if ( v23 )
                    v25 = (__int64 *)v23;
                }
LABEL_18:
                *v25 = v29;
                v25[1] = v19[1];
                v19[1] = 0;
                v27 = (unsigned int)(2 * (*(_DWORD *)(a1 + 8) >> 1) + 2);
                *(_DWORD *)(a1 + 8) = v27 | *(_DWORD *)(a1 + 8) & 1;
                v28 = v19[1];
                if ( v28 )
                {
                  v48 = v19[1];
                  sub_34A2530((unsigned int *)(v28 + 8), v20, v27, v24, v28, v23);
                  j_j___libc_free_0(v48);
                }
              }
              v19 += 2;
            }
            while ( v12 != v19 );
          }
          return (__int64 *)sub_C7D6A0(v4, v9, 8);
        }
LABEL_9:
        v13 = (_QWORD *)(a1 + 16);
        v14 = 8;
        goto LABEL_10;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v2 = 64;
        v8 = 1024;
        goto LABEL_5;
      }
      v15 = (__int64 *)(a1 + 16);
      v16 = (__int64 *)(a1 + 80);
      v2 = 64;
    }
  }
  v31 = v15;
  v32 = (__int64 *)v49;
  do
  {
    v33 = *v31;
    if ( *v31 != -4096 && v33 != -8192 )
    {
      if ( v32 )
        *v32 = v33;
      v32 += 2;
      *(v32 - 1) = v31[1];
    }
    v31 += 2;
  }
  while ( v31 != v16 );
  if ( v2 > 4 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v34 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v34;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v35 = *(__int64 **)(a1 + 16);
    v36 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v35 = v15;
    v36 = 8;
  }
  for ( j = &v35[v36]; j != v35; v35 += 2 )
  {
    if ( v35 )
      *v35 = -4096;
  }
  result = (__int64 *)v49;
  if ( v32 != (__int64 *)v49 )
  {
    do
    {
      v46 = *result;
      if ( *result != -8192 && v46 != -4096 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v39 = v15;
          v40 = 3;
        }
        else
        {
          v47 = *(_DWORD *)(a1 + 24);
          v39 = *(__int64 **)(a1 + 16);
          if ( !v47 )
          {
            MEMORY[0] = *result;
            BUG();
          }
          v40 = v47 - 1;
        }
        v41 = 1;
        v42 = 0;
        v43 = v40 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v44 = &v39[2 * v43];
        v45 = *v44;
        if ( v46 != *v44 )
        {
          while ( v45 != -4096 )
          {
            if ( v45 == -8192 && !v42 )
              v42 = v44;
            v43 = v40 & (v41 + v43);
            v44 = &v39[2 * v43];
            v45 = *v44;
            if ( v46 == *v44 )
              goto LABEL_46;
            ++v41;
          }
          if ( v42 )
            v44 = v42;
        }
LABEL_46:
        *v44 = v46;
        v44[1] = result[1];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      }
      result += 2;
    }
    while ( v32 != result );
  }
  return result;
}
