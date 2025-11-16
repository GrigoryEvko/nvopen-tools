// Function: sub_136EE70
// Address: 0x136ee70
//
__int64 __fastcall sub_136EE70(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // r15
  __int64 v8; // rax
  _DWORD *v9; // r12
  __int64 v10; // rdx
  _BYTE *v11; // rcx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // rdi
  int *v15; // r12
  __int64 v16; // r13
  bool v17; // zf
  int *v18; // rcx
  _DWORD *v19; // rax
  __int64 v20; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  unsigned int v23; // edi
  __int64 v24; // r10
  int v25; // r9d
  int v26; // r14d
  int *v27; // r13
  unsigned int v28; // esi
  int *v29; // rdx
  int v30; // r11d
  __int64 v31; // rsi
  int v32; // edx
  __int64 v33; // rax
  __int64 v34; // r8
  int v35; // edi
  int v36; // r14d
  int *v37; // r10
  unsigned int v38; // esi
  int *v39; // rax
  int v40; // r9d
  int v41; // edx
  int v42; // eax
  _BYTE v43[112]; // [rsp+10h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v15 = *(int **)(a1 + 16);
    v16 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
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
      v7 = 16LL * (unsigned int)v5;
      if ( v4 )
        goto LABEL_5;
      v15 = *(int **)(a1 + 16);
      v16 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 1024;
        v6 = 64;
LABEL_5:
        v8 = a1 + 16;
        v9 = v43;
        do
        {
          if ( *(_DWORD *)v8 <= 0xFFFFFFFD )
          {
            if ( v9 )
              *v9 = *(_DWORD *)v8;
            v9 += 4;
            *((_QWORD *)v9 - 1) = *(_QWORD *)(v8 + 8);
          }
          v8 += 16;
        }
        while ( v8 != a1 + 80 );
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v7);
        v10 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = result;
        v11 = v43;
        v12 = v10 & 1;
        *(_QWORD *)(a1 + 8) = v12;
        if ( (_BYTE)v12 )
        {
          result = a1 + 16;
          v7 = 64;
        }
        v13 = result;
        v14 = result + v7;
        while ( 1 )
        {
          if ( v13 )
            *(_DWORD *)result = -1;
          result += 16;
          if ( v14 == result )
            break;
          v13 = result;
        }
        while ( v9 != (_DWORD *)v11 )
        {
          v41 = *(_DWORD *)v11;
          if ( *(_DWORD *)v11 <= 0xFFFFFFFD )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v34 = a1 + 16;
              v35 = 3;
            }
            else
            {
              v42 = *(_DWORD *)(a1 + 24);
              v34 = *(_QWORD *)(a1 + 16);
              if ( !v42 )
                goto LABEL_72;
              v35 = v42 - 1;
            }
            v36 = 1;
            v37 = 0;
            v38 = v35 & (37 * v41);
            v39 = (int *)(v34 + 16LL * v38);
            v40 = *v39;
            if ( v41 != *v39 )
            {
              while ( v40 != -1 )
              {
                if ( v40 == -2 && !v37 )
                  v37 = v39;
                v38 = v35 & (v36 + v38);
                v39 = (int *)(v34 + 16LL * v38);
                v40 = *v39;
                if ( v41 == *v39 )
                  goto LABEL_44;
                ++v36;
              }
              if ( v37 )
                v39 = v37;
            }
LABEL_44:
            *v39 = v41;
            *((_QWORD *)v39 + 1) = *((_QWORD *)v11 + 1);
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
          v11 += 16;
        }
        return result;
      }
      v15 = *(int **)(a1 + 16);
      v16 = *(unsigned int *)(a1 + 24);
      v7 = 1024;
      v6 = 64;
    }
    v33 = sub_22077B0(v7);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v17 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v18 = &v15[4 * v16];
  if ( v17 )
  {
    v19 = *(_DWORD **)(a1 + 16);
    v20 = 4LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v19 = (_DWORD *)(a1 + 16);
    v20 = 16;
  }
  for ( i = &v19[v20]; i != v19; v19 += 4 )
  {
    if ( v19 )
      *v19 = -1;
  }
  for ( j = v15; v18 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v23 = *j;
      if ( (unsigned int)*j <= 0xFFFFFFFD )
        break;
      j += 4;
      if ( v18 == j )
        return j___libc_free_0(v15);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v24 = a1 + 16;
      v25 = 3;
    }
    else
    {
      v32 = *(_DWORD *)(a1 + 24);
      v24 = *(_QWORD *)(a1 + 16);
      if ( !v32 )
      {
LABEL_72:
        MEMORY[0] = 0;
        BUG();
      }
      v25 = v32 - 1;
    }
    v26 = 1;
    v27 = 0;
    v28 = v25 & (37 * v23);
    v29 = (int *)(v24 + 16LL * v28);
    v30 = *v29;
    if ( v23 != *v29 )
    {
      while ( v30 != -1 )
      {
        if ( v30 == -2 && !v27 )
          v27 = v29;
        v28 = v25 & (v26 + v28);
        v29 = (int *)(v24 + 16LL * v28);
        v30 = *v29;
        if ( v23 == *v29 )
          goto LABEL_32;
        ++v26;
      }
      if ( v27 )
        v29 = v27;
    }
LABEL_32:
    *v29 = v23;
    v31 = *((_QWORD *)j + 1);
    j += 4;
    *((_QWORD *)v29 + 1) = v31;
  }
  return j___libc_free_0(v15);
}
