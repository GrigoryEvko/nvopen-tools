// Function: sub_2B81830
// Address: 0x2b81830
//
__int64 __fastcall sub_2B81830(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r14d
  char v4; // cl
  unsigned __int64 v5; // rax
  __int64 v6; // r12
  __int64 v7; // r13
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  int *v11; // r11
  _DWORD *v12; // rax
  __int64 v13; // rdx
  _DWORD *k; // rdx
  int *m; // rdx
  int v16; // ecx
  __int64 v17; // r9
  int v18; // esi
  int v19; // r8d
  int v20; // r15d
  int *v21; // r14
  unsigned int n; // eax
  int *v23; // rdi
  int v24; // r10d
  int v25; // esi
  __int64 result; // rax
  _DWORD *v27; // r13
  _DWORD *v28; // rdx
  _DWORD *v29; // rax
  int *v30; // r12
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 i; // rcx
  int *v34; // rdx
  int v35; // ecx
  _DWORD *v36; // r8
  int v37; // edi
  int v38; // esi
  int v39; // r15d
  int *v40; // r14
  int j; // eax
  int *v42; // r10
  int v43; // r11d
  int v44; // eax
  int v45; // edi
  unsigned int v46; // eax
  _BYTE v47[112]; // [rsp+10h] [rbp-70h] BYREF

  v2 = a2;
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 8 )
  {
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v27 = (_DWORD *)(a1 + 16);
    v28 = (_DWORD *)(a1 + 80);
    goto LABEL_38;
  }
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
  v2 = v5;
  if ( (unsigned int)v5 > 0x40 )
  {
    v27 = (_DWORD *)(a1 + 16);
    v28 = (_DWORD *)(a1 + 80);
    if ( !v4 )
    {
      v6 = *(_QWORD *)(a1 + 16);
      v7 = *(unsigned int *)(a1 + 24);
      v8 = 8LL * (unsigned int)v5;
      goto LABEL_5;
    }
    goto LABEL_38;
  }
  if ( v4 )
  {
    v27 = (_DWORD *)(a1 + 16);
    v28 = (_DWORD *)(a1 + 80);
    v2 = 64;
LABEL_38:
    v29 = v27;
    v30 = (int *)v47;
    while ( 1 )
    {
      while ( *v29 == -1 )
      {
        if ( v29[1] != -1 )
          goto LABEL_40;
        v29 += 2;
        if ( v29 == v28 )
        {
LABEL_47:
          if ( v2 > 8 )
          {
            *(_BYTE *)(a1 + 8) &= ~1u;
            v31 = sub_C7D670(8LL * v2, 4);
            *(_DWORD *)(a1 + 24) = v2;
            *(_QWORD *)(a1 + 16) = v31;
          }
          v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
          *(_QWORD *)(a1 + 8) &= 1uLL;
          if ( v10 )
          {
            result = *(_QWORD *)(a1 + 16);
            v32 = 8LL * *(unsigned int *)(a1 + 24);
          }
          else
          {
            result = (__int64)v27;
            v32 = 64;
          }
          for ( i = result + v32; i != result; result += 8 )
          {
            if ( result )
            {
              *(_DWORD *)result = -1;
              *(_DWORD *)(result + 4) = -1;
            }
          }
          v34 = (int *)v47;
          if ( v30 == (int *)v47 )
            return result;
          while ( 2 )
          {
            v35 = *v34;
            if ( *v34 == -1 )
            {
              if ( v34[1] == -1 )
                goto LABEL_74;
            }
            else if ( v35 == -2 && v34[1] == -2 )
            {
              goto LABEL_74;
            }
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v36 = v27;
              v37 = 7;
            }
            else
            {
              v45 = *(_DWORD *)(a1 + 24);
              v36 = *(_DWORD **)(a1 + 16);
              if ( !v45 )
              {
LABEL_92:
                MEMORY[0] = 0;
                BUG();
              }
              v37 = v45 - 1;
            }
            v38 = v34[1];
            v39 = 1;
            v40 = 0;
            for ( j = v37
                    & (((0xBF58476D1CE4E5B9LL
                       * ((unsigned int)(37 * v38) | ((unsigned __int64)(unsigned int)(37 * v35) << 32))) >> 31)
                     ^ (756364221 * v38)); ; j = v37 & v44 )
            {
              v42 = &v36[2 * j];
              v43 = *v42;
              if ( v35 == *v42 && v38 == v42[1] )
                break;
              if ( v43 == -1 )
              {
                if ( v42[1] == -1 )
                {
                  if ( v40 )
                    v42 = v40;
                  break;
                }
              }
              else if ( v43 == -2 && v42[1] == -2 && !v40 )
              {
                v40 = &v36[2 * j];
              }
              v44 = v39 + j;
              ++v39;
            }
            *v42 = v35;
            v42[1] = v38;
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
LABEL_74:
            v34 += 2;
            if ( v30 == v34 )
              return result;
            continue;
          }
        }
      }
      if ( *v29 != -2 || v29[1] != -2 )
      {
LABEL_40:
        if ( v30 )
          *(_QWORD *)v30 = *(_QWORD *)v29;
        v30 += 2;
      }
      v29 += 2;
      if ( v29 == v28 )
        goto LABEL_47;
    }
  }
  v6 = *(_QWORD *)(a1 + 16);
  v7 = *(unsigned int *)(a1 + 24);
  v2 = 64;
  v8 = 512;
LABEL_5:
  v9 = sub_C7D670(v8, 4);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v11 = (int *)(v6 + 8 * v7);
  if ( v10 )
  {
    v12 = *(_DWORD **)(a1 + 16);
    v13 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v12 = (_DWORD *)(a1 + 16);
    v13 = 16;
  }
  for ( k = &v12[v13]; k != v12; v12 += 2 )
  {
    if ( v12 )
    {
      *v12 = -1;
      v12[1] = -1;
    }
  }
  for ( m = (int *)v6; v11 != m; m += 2 )
  {
    v16 = *m;
    if ( *m == -1 )
    {
      if ( m[1] != -1 )
        goto LABEL_17;
    }
    else if ( v16 != -2 || m[1] != -2 )
    {
LABEL_17:
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v17 = a1 + 16;
        v18 = 7;
      }
      else
      {
        v25 = *(_DWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        if ( !v25 )
          goto LABEL_92;
        v18 = v25 - 1;
      }
      v19 = m[1];
      v20 = 1;
      v21 = 0;
      for ( n = v18
              & (((0xBF58476D1CE4E5B9LL * ((unsigned int)(37 * v19) | ((unsigned __int64)(unsigned int)(37 * v16) << 32))) >> 31)
               ^ (756364221 * v19)); ; n = v18 & v46 )
      {
        v23 = (int *)(v17 + 8LL * n);
        v24 = *v23;
        if ( v16 == *v23 && v19 == v23[1] )
          break;
        if ( v24 == -1 )
        {
          if ( v23[1] == -1 )
          {
            if ( v21 )
              v23 = v21;
            break;
          }
        }
        else if ( v24 == -2 && v23[1] == -2 && !v21 )
        {
          v21 = (int *)(v17 + 8LL * n);
        }
        v46 = v20 + n;
        ++v20;
      }
      *v23 = v16;
      v23[1] = m[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return sub_C7D6A0(v6, 8 * v7, 4);
}
