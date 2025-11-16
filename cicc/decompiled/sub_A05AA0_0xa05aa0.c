// Function: sub_A05AA0
// Address: 0xa05aa0
//
__int64 *__fastcall sub_A05AA0(__int64 a1, unsigned int a2)
{
  __int64 *v3; // r13
  char v4; // dl
  unsigned __int64 v5; // rax
  unsigned int v6; // r15d
  __int64 v7; // r14
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r14
  bool v11; // zf
  __int64 *v12; // r15
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rbx
  _QWORD *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rdi
  __int64 v21; // r8
  int v22; // r10d
  _QWORD *v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rdi
  __int64 v26; // rax
  int v27; // esi
  _QWORD *v28; // rbx
  __int64 v29; // rax
  __int64 *v30; // r13
  __int64 *result; // rax
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rax
  _QWORD *v35; // rax
  __int64 v36; // rdx
  _QWORD *k; // rdx
  int v38; // edi
  _QWORD *v39; // r8
  int v40; // edi
  unsigned int v41; // esi
  _QWORD *v42; // rdx
  __int64 v43; // r9
  int v44; // r14d
  _QWORD *v45; // r10
  __int64 v46; // rcx
  __int64 v47; // rax
  __int64 v48; // [rsp+0h] [rbp-40h] BYREF
  __int64 v49; // [rsp+8h] [rbp-38h]
  _BYTE v50[48]; // [rsp+10h] [rbp-30h] BYREF

  v3 = *(__int64 **)(a1 + 16);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 > 1 )
  {
    v5 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
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
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      if ( !v4 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v8 = 16LL * (unsigned int)v5;
        goto LABEL_5;
      }
      v28 = (_QWORD *)(a1 + 16);
      if ( v3 == (__int64 *)-4096LL )
      {
        v33 = 16LL * (unsigned int)v5;
        v30 = &v48;
        goto LABEL_46;
      }
      if ( v3 == (__int64 *)-8192LL )
      {
        v30 = &v48;
      }
      else
      {
        v29 = *(_QWORD *)(a1 + 24);
        v48 = *(_QWORD *)(a1 + 16);
        v30 = (__int64 *)v50;
        v49 = v29;
      }
    }
    else
    {
      if ( !v4 )
      {
        v7 = *(unsigned int *)(a1 + 24);
        v6 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 2 * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = &v3[v10];
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 2;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = v3; v12 != j; j += 2 )
        {
          v26 = *j;
          if ( *j != -4096 && v26 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = (_QWORD *)(a1 + 16);
              v18 = 0;
              v19 = 0;
              v20 = a1 + 16;
            }
            else
            {
              v27 = *(_DWORD *)(a1 + 24);
              v20 = *(_QWORD *)(a1 + 16);
              if ( !v27 )
              {
                MEMORY[0] = *j;
                BUG();
              }
              v19 = (unsigned int)(v27 - 1);
              v18 = (unsigned int)v19 & (((unsigned int)v26 >> 4) ^ ((unsigned int)v26 >> 9));
              v17 = (_QWORD *)(v20 + 16 * v18);
            }
            v21 = *v17;
            v22 = 1;
            v23 = 0;
            if ( v26 != *v17 )
            {
              while ( v21 != -4096 )
              {
                if ( !v23 && v21 == -8192 )
                  v23 = v17;
                v18 = (unsigned int)v19 & (v22 + (_DWORD)v18);
                v17 = (_QWORD *)(v20 + 16LL * (unsigned int)v18);
                v21 = *v17;
                if ( v26 == *v17 )
                  goto LABEL_16;
                ++v22;
              }
              if ( v23 )
                v17 = v23;
            }
LABEL_16:
            *v17 = v26;
            v17[1] = j[1];
            j[1] = 0;
            v24 = (unsigned int)(2 * (*(_DWORD *)(a1 + 8) >> 1) + 2);
            *(_DWORD *)(a1 + 8) = v24 | *(_DWORD *)(a1 + 8) & 1;
            v25 = j[1];
            if ( v25 )
              sub_BA65D0(v25, v19, v24, v18, v21, v23);
          }
        }
        return (__int64 *)sub_C7D6A0(v3, v10 * 8, 8);
      }
      v28 = (_QWORD *)(a1 + 16);
      if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
      {
        v33 = 1024;
        v6 = 64;
        v30 = &v48;
        goto LABEL_46;
      }
      v32 = *(_QWORD *)(a1 + 24);
      v48 = *(_QWORD *)(a1 + 16);
      v30 = (__int64 *)v50;
      v6 = 64;
      v49 = v32;
    }
    v33 = 16LL * v6;
LABEL_46:
    *(_BYTE *)(a1 + 8) &= ~1u;
    v34 = sub_C7D670(v33, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v34;
    goto LABEL_47;
  }
  if ( !v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_6;
  }
  v28 = (_QWORD *)(a1 + 16);
  if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
  {
    v30 = &v48;
  }
  else
  {
    v47 = *(_QWORD *)(a1 + 24);
    v48 = *(_QWORD *)(a1 + 16);
    v30 = (__int64 *)v50;
    v49 = v47;
  }
LABEL_47:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v35 = *(_QWORD **)(a1 + 16);
    v36 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v35 = v28;
    v36 = 2;
  }
  for ( k = &v35[v36]; k != v35; v35 += 2 )
  {
    if ( v35 )
      *v35 = -4096;
  }
  for ( result = &v48; result != v30; result += 2 )
  {
    v46 = *result;
    if ( *result != -4096 && v46 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v42 = v28;
        v39 = v28;
        v41 = 0;
        v40 = 0;
      }
      else
      {
        v38 = *(_DWORD *)(a1 + 24);
        v39 = *(_QWORD **)(a1 + 16);
        if ( !v38 )
        {
          MEMORY[0] = *result;
          BUG();
        }
        v40 = v38 - 1;
        v41 = v40 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
        v42 = &v39[2 * v41];
      }
      v43 = *v42;
      v44 = 1;
      v45 = 0;
      if ( v46 != *v42 )
      {
        while ( v43 != -4096 )
        {
          if ( !v45 && v43 == -8192 )
            v45 = v42;
          v41 = v40 & (v44 + v41);
          v42 = &v39[2 * v41];
          v43 = *v42;
          if ( v46 == *v42 )
            goto LABEL_58;
          ++v44;
        }
        if ( v45 )
          v42 = v45;
      }
LABEL_58:
      *v42 = v46;
      v42[1] = result[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
