// Function: sub_2D72880
// Address: 0x2d72880
//
__int64 *__fastcall sub_2D72880(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r15d
  __int64 v4; // r12
  char v5; // r13
  unsigned int v6; // eax
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // r9
  bool v11; // zf
  __int64 *v12; // r8
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v17; // r10
  int v18; // edi
  int v19; // r15d
  _QWORD *v20; // r14
  unsigned int v21; // esi
  _QWORD *v22; // rcx
  __int64 v23; // r11
  __int64 v24; // rdx
  int v25; // edi
  __int64 *result; // rax
  __int64 *v27; // r14
  __int64 *v28; // rcx
  __int64 *v29; // rax
  __int64 *v30; // r12
  __int64 v31; // rdx
  __int64 *v32; // rax
  __int64 v33; // rdx
  __int64 *k; // rdx
  __int64 *v35; // r8
  int v36; // edi
  int v37; // r13d
  __int64 *v38; // r10
  unsigned int v39; // esi
  __int64 *v40; // rcx
  __int64 v41; // r9
  int v42; // edi
  __int64 v43; // rax
  _BYTE v44[560]; // [rsp+10h] [rbp-230h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0x20 )
  {
    if ( !v5 )
    {
      v7 = *(_DWORD *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v27 = (__int64 *)(a1 + 16);
    v28 = (__int64 *)(a1 + 528);
  }
  else
  {
    v6 = sub_AF1560(a2 - 1);
    v2 = v6;
    if ( v6 > 0x40 )
    {
      v27 = (__int64 *)(a1 + 16);
      v28 = (__int64 *)(a1 + 528);
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * v6;
        goto LABEL_5;
      }
    }
    else
    {
      if ( !v5 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 1024;
        v2 = 64;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v2;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
        v10 = 16LL * v7;
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        v12 = (__int64 *)(v4 + v10);
        if ( v11 )
        {
          v13 = *(_QWORD **)(a1 + 16);
          v14 = 2LL * *(unsigned int *)(a1 + 24);
        }
        else
        {
          v13 = (_QWORD *)(a1 + 16);
          v14 = 64;
        }
        for ( i = &v13[v14]; i != v13; v13 += 2 )
        {
          if ( v13 )
            *v13 = -4096;
        }
        for ( j = (__int64 *)v4; v12 != j; j += 2 )
        {
          v24 = *j;
          if ( *j != -4096 && v24 != -8192 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v17 = a1 + 16;
              v18 = 31;
            }
            else
            {
              v25 = *(_DWORD *)(a1 + 24);
              v17 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
                goto LABEL_77;
              v18 = v25 - 1;
            }
            v19 = 1;
            v20 = 0;
            v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v22 = (_QWORD *)(v17 + 16LL * v21);
            v23 = *v22;
            if ( *v22 != v24 )
            {
              while ( v23 != -4096 )
              {
                if ( v23 == -8192 && !v20 )
                  v20 = v22;
                v21 = v18 & (v19 + v21);
                v22 = (_QWORD *)(v17 + 16LL * v21);
                v23 = *v22;
                if ( v24 == *v22 )
                  goto LABEL_18;
                ++v19;
              }
              if ( v20 )
                v22 = v20;
            }
LABEL_18:
            *v22 = v24;
            v22[1] = j[1];
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return (__int64 *)sub_C7D6A0(v4, v10, 8);
      }
      v27 = (__int64 *)(a1 + 16);
      v28 = (__int64 *)(a1 + 528);
      v2 = 64;
    }
  }
  v29 = v27;
  v30 = (__int64 *)v44;
  do
  {
    v31 = *v29;
    if ( *v29 != -4096 && v31 != -8192 )
    {
      if ( v30 )
        *v30 = v31;
      v30 += 2;
      *(v30 - 1) = v29[1];
    }
    v29 += 2;
  }
  while ( v29 != v28 );
  if ( v2 > 0x20 )
  {
    *(_BYTE *)(a1 + 8) &= ~1u;
    v43 = sub_C7D670(16LL * v2, 8);
    *(_DWORD *)(a1 + 24) = v2;
    *(_QWORD *)(a1 + 16) = v43;
  }
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v32 = *(__int64 **)(a1 + 16);
    v33 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v32 = v27;
    v33 = 64;
  }
  for ( k = &v32[v33]; k != v32; v32 += 2 )
  {
    if ( v32 )
      *v32 = -4096;
  }
  result = (__int64 *)v44;
  if ( v30 != (__int64 *)v44 )
  {
    do
    {
      v24 = *result;
      if ( *result != -4096 && v24 != -8192 )
      {
        if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
        {
          v35 = v27;
          v36 = 31;
        }
        else
        {
          v42 = *(_DWORD *)(a1 + 24);
          v35 = *(__int64 **)(a1 + 16);
          if ( !v42 )
          {
LABEL_77:
            MEMORY[0] = v24;
            BUG();
          }
          v36 = v42 - 1;
        }
        v37 = 1;
        v38 = 0;
        v39 = v36 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
        v40 = &v35[2 * v39];
        v41 = *v40;
        if ( v24 != *v40 )
        {
          while ( v41 != -4096 )
          {
            if ( v41 == -8192 && !v38 )
              v38 = v40;
            v39 = v36 & (v37 + v39);
            v40 = &v35[2 * v39];
            v41 = *v40;
            if ( v24 == *v40 )
              goto LABEL_47;
            ++v37;
          }
          if ( v38 )
            v40 = v38;
        }
LABEL_47:
        *v40 = v24;
        v40[1] = result[1];
        *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      }
      result += 2;
    }
    while ( v30 != result );
  }
  return result;
}
