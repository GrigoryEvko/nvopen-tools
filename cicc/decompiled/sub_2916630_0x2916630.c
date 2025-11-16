// Function: sub_2916630
// Address: 0x2916630
//
__int64 *__fastcall sub_2916630(__int64 a1, unsigned int a2)
{
  unsigned int v2; // ecx
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rdi
  __int64 *v9; // r15
  __int64 *v10; // rsi
  __int64 v11; // r13
  bool v12; // zf
  __int64 *v13; // r15
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v18; // rdi
  int v19; // esi
  int v20; // r10d
  _QWORD *v21; // r9
  unsigned int v22; // ecx
  _QWORD *v23; // rdx
  __int64 v24; // r8
  unsigned __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // edx
  __int64 *v28; // rax
  __int64 *v29; // r13
  __int64 v30; // rdx
  __int64 *v31; // rax
  __int64 v32; // rdx
  __int64 *k; // rdx
  __int64 *result; // rax
  __int64 *v35; // r8
  int v36; // edi
  int v37; // r11d
  __int64 *v38; // r10
  unsigned int v39; // esi
  __int64 *v40; // rcx
  __int64 v41; // r9
  __int64 v42; // rdx
  int v43; // ecx
  unsigned int v44; // [rsp+Ch] [rbp-174h]
  unsigned int v45; // [rsp+Ch] [rbp-174h]
  _BYTE v46[368]; // [rsp+10h] [rbp-170h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    v9 = (__int64 *)(a1 + 16);
    v10 = (__int64 *)(a1 + 336);
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
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
      v9 = (__int64 *)(a1 + 16);
      v10 = (__int64 *)(a1 + 336);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 40LL * (unsigned int)v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v2 = 64;
        v8 = 2560;
LABEL_5:
        v44 = v2;
        *(_QWORD *)(a1 + 16) = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v44;
LABEL_8:
        v11 = 40LL * v7;
        v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v13 = (__int64 *)(v4 + v11);
        if ( v12 )
        {
          v14 = *(_QWORD **)(a1 + 16);
          v15 = 5LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v14 = (_QWORD *)(a1 + 16);
          v15 = 40;
        }
        for ( i = &v14[v15]; i != v14; v14 += 5 )
        {
          if ( v14 )
            *v14 = -4096;
        }
        for ( j = (__int64 *)v4; v13 != j; j += 5 )
        {
          v26 = *j;
          if ( *j != -4096 && v26 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v18 = a1 + 16;
              v19 = 7;
            }
            else
            {
              v27 = *(_DWORD *)(a1 + 24);
              v18 = *(_QWORD *)(a1 + 16);
              if ( !v27 )
              {
                MEMORY[0] = *j;
                BUG();
              }
              v19 = v27 - 1;
            }
            v20 = 1;
            v21 = 0;
            v22 = v19 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
            v23 = (_QWORD *)(v18 + 40LL * v22);
            v24 = *v23;
            if ( *v23 != v26 )
            {
              while ( v24 != -4096 )
              {
                if ( !v21 && v24 == -8192 )
                  v21 = v23;
                v22 = v19 & (v20 + v22);
                v23 = (_QWORD *)(v18 + 40LL * v22);
                v24 = *v23;
                if ( v26 == *v23 )
                  goto LABEL_18;
                ++v20;
              }
              if ( v21 )
                v23 = v21;
            }
LABEL_18:
            *v23 = v26;
            v23[1] = j[1];
            v23[2] = j[2];
            v23[3] = j[3];
            v23[4] = j[4];
            j[2] = 0;
            j[4] = 0;
            j[3] = 0;
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            v25 = j[2];
            if ( v25 )
              j_j___libc_free_0(v25);
          }
        }
        return (__int64 *)sub_C7D6A0(v4, v11, 8);
      }
      v9 = (__int64 *)(a1 + 16);
      v10 = (__int64 *)(a1 + 336);
      v2 = 64;
    }
  }
  v28 = v9;
  v29 = (__int64 *)v46;
  do
  {
    v30 = *v28;
    if ( *v28 != -4096 && v30 != -8192 )
    {
      if ( v29 )
        *v29 = v30;
      v29 += 5;
      *(v29 - 4) = v28[1];
      *(v29 - 3) = v28[2];
      *(v29 - 2) = v28[3];
      *(v29 - 1) = v28[4];
    }
    v28 += 5;
  }
  while ( v28 != v10 );
  if ( v2 > 8 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v45 = v2;
    *(_QWORD *)(a1 + 16) = sub_C7D670(40LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v45;
  }
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v12 )
  {
    v31 = *(__int64 **)(a1 + 16);
    v32 = 5LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v31 = v9;
    v32 = 40;
  }
  for ( k = &v31[v32]; k != v31; v31 += 5 )
  {
    if ( v31 )
      *v31 = -4096;
  }
  result = (__int64 *)v46;
  if ( v29 != (__int64 *)v46 )
  {
    do
    {
      v42 = *result;
      if ( *result != -8192 && v42 != -4096 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v35 = v9;
          v36 = 7;
        }
        else
        {
          v43 = *(_DWORD *)(a1 + 24);
          v35 = *(__int64 **)(a1 + 16);
          if ( !v43 )
          {
            MEMORY[0] = *result;
            BUG();
          }
          v36 = v43 - 1;
        }
        v37 = 1;
        v38 = 0;
        v39 = v36 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v40 = &v35[5 * v39];
        v41 = *v40;
        if ( v42 != *v40 )
        {
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v38 )
              v38 = v40;
            v39 = v36 & (v37 + v39);
            v40 = &v35[5 * v39];
            v41 = *v40;
            if ( v42 == *v40 )
              goto LABEL_46;
            ++v37;
          }
          if ( v38 )
            v40 = v38;
        }
LABEL_46:
        *v40 = v42;
        v40[1] = result[1];
        v40[2] = result[2];
        v40[3] = result[3];
        v40[4] = result[4];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      }
      result += 5;
    }
    while ( v29 != result );
  }
  return result;
}
