// Function: sub_15177F0
// Address: 0x15177f0
//
__int64 __fastcall sub_15177F0(__int64 a1, int a2)
{
  __int64 result; // rax
  char v4; // cl
  unsigned __int64 v5; // rax
  int v6; // r15d
  __int64 v7; // r13
  unsigned int *v8; // r12
  bool v9; // zf
  __int64 v10; // rdx
  __int64 v11; // rdi
  unsigned int *v12; // rdx
  int v13; // edi
  __int64 v14; // r9
  int v15; // edi
  unsigned int v16; // esi
  _DWORD *v17; // rcx
  int v18; // r8d
  int v19; // r11d
  _DWORD *v20; // r10
  int *v21; // r12
  __int64 v22; // r14
  int *v23; // rcx
  _DWORD *v24; // rax
  __int64 v25; // rdx
  _DWORD *i; // rdx
  int *j; // rax
  int v28; // edx
  _DWORD *v29; // rsi
  unsigned int v30; // r8d
  int v31; // edi
  __int64 v32; // r10
  int v33; // r9d
  int v34; // r13d
  _DWORD *v35; // r11
  int v36; // edi
  __int64 v37; // rax
  int v38; // [rsp+1Ch] [rbp-34h] BYREF
  char v39; // [rsp+20h] [rbp-30h] BYREF

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
      v7 = 4LL * (unsigned int)v5;
      if ( v4 )
      {
LABEL_5:
        v8 = (unsigned int *)&v38;
        if ( *(_DWORD *)(a1 + 16) <= 0xFFFFFFFD )
        {
          v38 = *(_DWORD *)(a1 + 16);
          v8 = (unsigned int *)&v39;
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
          v7 = 4;
        }
        v11 = result + v7;
        while ( 1 )
        {
          if ( v10 )
            *(_DWORD *)result = -1;
          result += 4;
          if ( v11 == result )
            break;
          v10 = result;
        }
        v12 = (unsigned int *)&v38;
        if ( v8 != (unsigned int *)&v38 )
        {
          do
          {
            while ( 1 )
            {
              result = *v12;
              if ( (unsigned int)result <= 0xFFFFFFFD )
                break;
              if ( ++v12 == v8 )
                return result;
            }
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = (_DWORD *)(a1 + 16);
              v14 = a1 + 16;
              v16 = 0;
              v15 = 0;
            }
            else
            {
              v13 = *(_DWORD *)(a1 + 24);
              v14 = *(_QWORD *)(a1 + 16);
              if ( !v13 )
                goto LABEL_67;
              v15 = v13 - 1;
              v16 = v15 & (37 * result);
              v17 = (_DWORD *)(v14 + 4LL * v16);
            }
            v18 = *v17;
            v19 = 1;
            v20 = 0;
            if ( (_DWORD)result != *v17 )
            {
              while ( v18 != -1 )
              {
                if ( v18 == -2 && !v20 )
                  v20 = v17;
                v16 = v15 & (v19 + v16);
                v17 = (_DWORD *)(v14 + 4LL * v16);
                v18 = *v17;
                if ( (_DWORD)result == *v17 )
                  goto LABEL_22;
                ++v19;
              }
              if ( v20 )
                v17 = v20;
            }
LABEL_22:
            *v17 = result;
            ++v12;
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
          while ( v12 != v8 );
        }
        return result;
      }
      v21 = *(int **)(a1 + 16);
      v22 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 256;
        v6 = 64;
        goto LABEL_5;
      }
      v21 = *(int **)(a1 + 16);
      v22 = *(unsigned int *)(a1 + 24);
      v7 = 256;
      v6 = 64;
    }
    v37 = sub_22077B0(v7);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v37;
  }
  else
  {
    if ( v4 )
      return result;
    v21 = *(int **)(a1 + 16);
    v22 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  v9 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v23 = &v21[v22];
  if ( v9 )
  {
    v24 = *(_DWORD **)(a1 + 16);
    v25 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v24 = (_DWORD *)(a1 + 16);
    v25 = 1;
  }
  for ( i = &v24[v25]; i != v24; ++v24 )
  {
    if ( v24 )
      *v24 = -1;
  }
  for ( j = v21; v23 != j; *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1 )
  {
    while ( 1 )
    {
      v28 = *j;
      if ( (unsigned int)*j <= 0xFFFFFFFD )
        break;
      if ( v23 == ++j )
        return j___libc_free_0(v21);
    }
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v29 = (_DWORD *)(a1 + 16);
      v30 = 0;
      v31 = 0;
      v32 = a1 + 16;
    }
    else
    {
      v36 = *(_DWORD *)(a1 + 24);
      v32 = *(_QWORD *)(a1 + 16);
      if ( !v36 )
      {
LABEL_67:
        MEMORY[0] = 0;
        BUG();
      }
      v31 = v36 - 1;
      v30 = v31 & (37 * v28);
      v29 = (_DWORD *)(v32 + 4LL * v30);
    }
    v33 = *v29;
    v34 = 1;
    v35 = 0;
    if ( v28 != *v29 )
    {
      while ( v33 != -1 )
      {
        if ( v33 == -2 && !v35 )
          v35 = v29;
        v30 = v31 & (v34 + v30);
        v29 = (_DWORD *)(v32 + 4LL * v30);
        v33 = *v29;
        if ( v28 == *v29 )
          goto LABEL_39;
        ++v34;
      }
      if ( v35 )
        v29 = v35;
    }
LABEL_39:
    *v29 = v28;
    ++j;
  }
  return j___libc_free_0(v21);
}
