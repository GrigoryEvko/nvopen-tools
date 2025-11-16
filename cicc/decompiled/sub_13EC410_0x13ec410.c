// Function: sub_13EC410
// Address: 0x13ec410
//
__int64 __fastcall sub_13EC410(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 v5; // r13
  unsigned __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // r14
  __int64 *v9; // r8
  unsigned int *v10; // rbx
  unsigned int *v11; // r13
  __int64 v12; // rax
  int *v13; // rdi
  bool v14; // zf
  __int64 v15; // rcx
  __int64 v16; // r14
  unsigned int *v17; // rbx
  __int64 v18; // r8
  int v19; // r9d
  int v20; // r11d
  __int64 *v21; // r10
  unsigned int v22; // ecx
  __int64 *v23; // rdi
  __int64 v24; // rsi
  __int64 v25; // rax
  int *v26; // rdi
  int v27; // ecx
  unsigned int v28; // ebx
  __int64 v29; // r14
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v34; // rsi
  int v35; // r8d
  int v36; // r10d
  __int64 *v37; // r9
  unsigned int v38; // edx
  __int64 *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rax
  int *v42; // rdi
  int v43; // edx
  __int64 v44; // rdi
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rdi
  __int64 v48; // rdi
  __int64 v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // [rsp+10h] [rbp-100h]
  __int64 *v52; // [rsp+18h] [rbp-F8h]
  __int64 *v53; // [rsp+18h] [rbp-F8h]
  _BYTE v54[240]; // [rsp+20h] [rbp-F0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v5 = *(_QWORD *)(a1 + 16);
    v28 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = *(_QWORD *)(a1 + 16);
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
    v7 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v8 = 48LL * (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v28 = *(_DWORD *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 3072;
        v7 = 64;
LABEL_5:
        v9 = (__int64 *)(a1 + 208);
        v51 = a1 + 16;
        v10 = (unsigned int *)(a1 + 16);
        v11 = (unsigned int *)v54;
        do
        {
          v12 = *(_QWORD *)v10;
          if ( *(_QWORD *)v10 != -8 && v12 != -16 )
          {
            if ( v11 )
              *(_QWORD *)v11 = v12;
            v11[2] = 0;
            v13 = (int *)(v11 + 2);
            v11 += 12;
            v52 = v9;
            sub_13E8810(v13, v10 + 2);
            v9 = v52;
            if ( v10[2] == 3 )
            {
              if ( v10[10] > 0x40 )
              {
                v49 = *((_QWORD *)v10 + 4);
                if ( v49 )
                {
                  j_j___libc_free_0_0(v49);
                  v9 = v52;
                }
              }
              if ( v10[6] > 0x40 )
              {
                v50 = *((_QWORD *)v10 + 2);
                if ( v50 )
                {
                  v53 = v9;
                  j_j___libc_free_0_0(v50);
                  v9 = v53;
                }
              }
            }
          }
          v10 += 12;
        }
        while ( v10 != (unsigned int *)v9 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v8);
        v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = result;
        v15 = result;
        *(_DWORD *)(a1 + 24) = v7;
        if ( !v14 )
        {
          result = v51;
          v8 = 192;
          v15 = v51;
        }
        v16 = result + v8;
        while ( 1 )
        {
          if ( v15 )
            *(_QWORD *)result = -8;
          result += 48;
          if ( v16 == result )
            break;
          v15 = result;
        }
        v17 = (unsigned int *)v54;
        if ( v11 != (unsigned int *)v54 )
        {
          while ( 1 )
          {
            result = *(_QWORD *)v17;
            if ( *(_QWORD *)v17 != -16 && result != -8 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v18 = v51;
                v19 = 3;
              }
              else
              {
                v27 = *(_DWORD *)(a1 + 24);
                v18 = *(_QWORD *)(a1 + 16);
                if ( !v27 )
                  goto LABEL_92;
                v19 = v27 - 1;
              }
              v20 = 1;
              v21 = 0;
              v22 = v19 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
              v23 = (__int64 *)(v18 + 48LL * v22);
              v24 = *v23;
              if ( result != *v23 )
              {
                while ( v24 != -8 )
                {
                  if ( v24 == -16 && !v21 )
                    v21 = v23;
                  v22 = v19 & (v20 + v22);
                  v23 = (__int64 *)(v18 + 48LL * v22);
                  v24 = *v23;
                  if ( result == *v23 )
                    goto LABEL_23;
                  ++v20;
                }
                if ( v21 )
                  v23 = v21;
              }
LABEL_23:
              v25 = *(_QWORD *)v17;
              *((_DWORD *)v23 + 2) = 0;
              v26 = (int *)(v23 + 1);
              *((_QWORD *)v26 - 1) = v25;
              sub_13E8810(v26, v17 + 2);
              result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
              v14 = v17[2] == 3;
              *(_DWORD *)(a1 + 8) = result;
              if ( v14 )
              {
                if ( v17[10] > 0x40 )
                {
                  v47 = *((_QWORD *)v17 + 4);
                  if ( v47 )
                    result = j_j___libc_free_0_0(v47);
                }
                if ( v17[6] > 0x40 )
                {
                  v48 = *((_QWORD *)v17 + 2);
                  if ( v48 )
                    result = j_j___libc_free_0_0(v48);
                }
              }
            }
            v17 += 12;
            if ( v11 == v17 )
              return result;
          }
        }
        return result;
      }
      v28 = *(_DWORD *)(a1 + 24);
      v8 = 3072;
      v7 = 64;
    }
    v46 = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v46;
  }
  v29 = v5 + 48LL * v28;
  v14 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v14 )
  {
    v30 = *(_QWORD **)(a1 + 16);
    v31 = 6LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = (_QWORD *)(a1 + 16);
    v31 = 24;
  }
  for ( i = &v30[v31]; i != v30; v30 += 6 )
  {
    if ( v30 )
      *v30 = -8;
  }
  for ( j = v5; v29 != j; j += 48 )
  {
    result = *(_QWORD *)j;
    if ( *(_QWORD *)j != -16 && result != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v34 = a1 + 16;
        v35 = 3;
      }
      else
      {
        v43 = *(_DWORD *)(a1 + 24);
        v34 = *(_QWORD *)(a1 + 16);
        if ( !v43 )
        {
LABEL_92:
          MEMORY[0] = result;
          BUG();
        }
        v35 = v43 - 1;
      }
      v36 = 1;
      v37 = 0;
      v38 = v35 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
      v39 = (__int64 *)(v34 + 48LL * v38);
      v40 = *v39;
      if ( result != *v39 )
      {
        while ( v40 != -8 )
        {
          if ( !v37 && v40 == -16 )
            v37 = v39;
          v38 = v35 & (v36 + v38);
          v39 = (__int64 *)(v34 + 48LL * v38);
          v40 = *v39;
          if ( result == *v39 )
            goto LABEL_43;
          ++v36;
        }
        if ( v37 )
          v39 = v37;
      }
LABEL_43:
      v41 = *(_QWORD *)j;
      *((_DWORD *)v39 + 2) = 0;
      v42 = (int *)(v39 + 1);
      *((_QWORD *)v42 - 1) = v41;
      sub_13E8810(v42, (unsigned int *)(j + 8));
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      if ( *(_DWORD *)(j + 8) == 3 )
      {
        if ( *(_DWORD *)(j + 40) > 0x40u )
        {
          v44 = *(_QWORD *)(j + 32);
          if ( v44 )
            j_j___libc_free_0_0(v44);
        }
        if ( *(_DWORD *)(j + 24) > 0x40u )
        {
          v45 = *(_QWORD *)(j + 16);
          if ( v45 )
            j_j___libc_free_0_0(v45);
        }
      }
    }
  }
  return j___libc_free_0(v5);
}
