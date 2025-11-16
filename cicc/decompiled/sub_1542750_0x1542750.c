// Function: sub_1542750
// Address: 0x1542750
//
__int64 __fastcall sub_1542750(__int64 a1, int a2)
{
  __int64 result; // rax
  char v4; // cl
  unsigned __int64 v5; // rax
  int v6; // r15d
  __int64 v7; // r14
  int *v8; // r12
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rdi
  int *i; // rdx
  int v13; // ecx
  int v14; // esi
  __int64 v15; // r9
  int v16; // esi
  unsigned int v17; // edi
  int *v18; // rax
  int v19; // r8d
  int v20; // r11d
  int *v21; // r10
  __int64 v22; // rcx
  int *v23; // r13
  __int64 v24; // r12
  int *v25; // r12
  _DWORD *v26; // rax
  __int64 v27; // rdx
  _DWORD *j; // rdx
  int *k; // rax
  unsigned int v30; // esi
  unsigned int v31; // ecx
  unsigned int *v32; // rdx
  __int64 v33; // r9
  int v34; // edi
  __int64 v35; // r8
  int v36; // r11d
  unsigned int *v37; // r10
  __int64 v38; // rcx
  int v39; // edx
  __int64 v40; // rax
  int v41; // [rsp+10h] [rbp-40h] BYREF
  __int64 v42; // [rsp+14h] [rbp-3Ch]
  int v43; // [rsp+1Ch] [rbp-34h]
  char v44; // [rsp+20h] [rbp-30h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 )
  {
    v5 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
              | (unsigned int)(a2 - 1)
              | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
            | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
          | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 16)
        | (((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
            | (unsigned int)(a2 - 1)
            | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
          | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 8)
        | (((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
          | (unsigned int)(a2 - 1)
          | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 4)
        | (((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
        | (unsigned int)(a2 - 1)
        | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1))
       + 1;
    v6 = v5;
    if ( (unsigned int)v5 > 0x40 )
    {
      v7 = 16LL * (unsigned int)v5;
      if ( v4 )
      {
LABEL_5:
        v8 = &v41;
        if ( *(_DWORD *)(a1 + 16) <= 0xFFFFFFFD )
        {
          v41 = *(_DWORD *)(a1 + 16);
          v8 = (int *)&v44;
          v42 = *(_QWORD *)(a1 + 20);
          v43 = *(_DWORD *)(a1 + 28);
        }
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v7);
        v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = result;
        v10 = result;
        *(_DWORD *)(a1 + 24) = v6;
        if ( !v9 )
        {
          v10 = a1 + 16;
          result = a1 + 16;
          v7 = 16;
        }
        v11 = result + v7;
        while ( 1 )
        {
          if ( v10 )
            *(_DWORD *)result = -1;
          result += 16;
          if ( v11 == result )
            break;
          v10 = result;
        }
        for ( i = &v41; v8 != i; *(_DWORD *)(a1 + 8) = result )
        {
          while ( 1 )
          {
            v13 = *i;
            if ( (unsigned int)*i <= 0xFFFFFFFD )
              break;
            i += 4;
            if ( v8 == i )
              return result;
          }
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v18 = (int *)(a1 + 16);
            v15 = a1 + 16;
            v17 = 0;
            v16 = 0;
          }
          else
          {
            v14 = *(_DWORD *)(a1 + 24);
            v15 = *(_QWORD *)(a1 + 16);
            if ( !v14 )
              goto LABEL_66;
            v16 = v14 - 1;
            v17 = v16 & (37 * v13);
            v18 = (int *)(v15 + 16LL * v17);
          }
          v19 = *v18;
          v20 = 1;
          v21 = 0;
          if ( v13 != *v18 )
          {
            while ( v19 != -1 )
            {
              if ( v19 == -2 && !v21 )
                v21 = v18;
              v17 = v16 & (v20 + v17);
              v18 = (int *)(v15 + 16LL * v17);
              v19 = *v18;
              if ( v13 == *v18 )
                goto LABEL_22;
              ++v20;
            }
            if ( v21 )
              v18 = v21;
          }
LABEL_22:
          *v18 = v13;
          v22 = *(_QWORD *)(i + 1);
          i += 4;
          *(_QWORD *)(v18 + 1) = v22;
          v18[3] = *(i - 1);
          result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
        }
        return result;
      }
      v23 = *(int **)(a1 + 16);
      v24 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 1024;
        v6 = 64;
        goto LABEL_5;
      }
      v23 = *(int **)(a1 + 16);
      v24 = *(unsigned int *)(a1 + 24);
      v7 = 1024;
      v6 = 64;
    }
    v40 = sub_22077B0(v7);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v40;
  }
  else
  {
    if ( v4 )
      return result;
    v23 = *(int **)(a1 + 16);
    v24 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  v25 = &v23[4 * v24];
  v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v9 )
  {
    v26 = *(_DWORD **)(a1 + 16);
    v27 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v26 = (_DWORD *)(a1 + 16);
    v27 = 4;
  }
  for ( j = &v26[v27]; j != v26; v26 += 4 )
  {
    if ( v26 )
      *v26 = -1;
  }
  for ( k = v23; v25 != k; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v30 = *k;
      if ( (unsigned int)*k <= 0xFFFFFFFD )
        break;
      k += 4;
      if ( v25 == k )
        return j___libc_free_0(v23);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v31 = *(_DWORD *)(a1 + 16);
      v32 = (unsigned int *)(a1 + 16);
      LODWORD(v33) = 0;
      v34 = 0;
      v35 = a1 + 16;
      v36 = 1;
      v37 = 0;
      if ( v30 != v31 )
        goto LABEL_42;
    }
    else
    {
      v39 = *(_DWORD *)(a1 + 24);
      v35 = *(_QWORD *)(a1 + 16);
      if ( !v39 )
      {
LABEL_66:
        MEMORY[0] = 0;
        BUG();
      }
      v34 = v39 - 1;
      v36 = 1;
      v37 = 0;
      v33 = (v39 - 1) & (37 * v30);
      v32 = (unsigned int *)(v35 + 16 * v33);
      v31 = *v32;
      if ( v30 != *v32 )
      {
LABEL_42:
        while ( v31 != -1 )
        {
          if ( v31 == -2 && !v37 )
            v37 = v32;
          v33 = v34 & (unsigned int)(v33 + v36);
          v32 = (unsigned int *)(v35 + 16 * v33);
          v31 = *v32;
          if ( v30 == *v32 )
            goto LABEL_38;
          ++v36;
        }
        if ( v37 )
          v32 = v37;
      }
    }
LABEL_38:
    *v32 = v30;
    v38 = *(_QWORD *)(k + 1);
    k += 4;
    *(_QWORD *)(v32 + 1) = v38;
    v32[3] = *(k - 1);
  }
  return j___libc_free_0(v23);
}
