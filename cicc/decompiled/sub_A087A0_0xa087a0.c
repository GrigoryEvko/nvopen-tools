// Function: sub_A087A0
// Address: 0xa087a0
//
__int64 *__fastcall sub_A087A0(__int64 a1, unsigned int a2)
{
  __int64 *v3; // r12
  char v4; // dl
  unsigned __int64 v5; // rax
  unsigned int v6; // r14d
  unsigned int v7; // r13d
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rsi
  bool v11; // zf
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  __int64 v14; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  _QWORD *v17; // rcx
  unsigned int v18; // r8d
  int v19; // r9d
  __int64 v20; // r10
  __int64 v21; // r11
  int v22; // r14d
  _QWORD *v23; // r13
  __int64 v24; // rdx
  int v25; // r9d
  __int64 v26; // rax
  __int64 *v27; // r12
  __int64 *result; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rax
  _QWORD *v32; // rax
  __int64 v33; // rdx
  _QWORD *k; // rdx
  _QWORD *v35; // rdx
  unsigned int v36; // esi
  int v37; // edi
  __int64 v38; // r8
  __int64 v39; // r9
  int v40; // r11d
  _QWORD *v41; // r10
  __int64 v42; // rcx
  int v43; // edi
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-40h] BYREF
  __int64 v46; // [rsp+8h] [rbp-38h]
  _BYTE v47[48]; // [rsp+10h] [rbp-30h] BYREF

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
        v7 = *(_DWORD *)(a1 + 24);
        v8 = 16LL * (unsigned int)v5;
        goto LABEL_5;
      }
      if ( v3 == (__int64 *)-4096LL )
      {
        v30 = 16LL * (unsigned int)v5;
        v27 = &v45;
        goto LABEL_45;
      }
      if ( v3 == (__int64 *)-8192LL )
      {
        v27 = &v45;
      }
      else
      {
        v26 = *(_QWORD *)(a1 + 24);
        v45 = *(_QWORD *)(a1 + 16);
        v27 = (__int64 *)v47;
        v46 = v26;
      }
    }
    else
    {
      if ( !v4 )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v6 = 64;
        v8 = 1024;
LABEL_5:
        v9 = sub_C7D670(v8, 8);
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 16) = v9;
LABEL_6:
        v10 = 2LL * v7;
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
          v24 = *j;
          if ( *j != -8192 && v24 != -4096 )
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
              v25 = *(_DWORD *)(a1 + 24);
              v20 = *(_QWORD *)(a1 + 16);
              if ( !v25 )
              {
                MEMORY[0] = *j;
                BUG();
              }
              v19 = v25 - 1;
              v18 = v19 & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
              v17 = (_QWORD *)(v20 + 16LL * v18);
            }
            v21 = *v17;
            v22 = 1;
            v23 = 0;
            if ( v24 != *v17 )
            {
              while ( v21 != -4096 )
              {
                if ( v21 == -8192 && !v23 )
                  v23 = v17;
                v18 = v19 & (v22 + v18);
                v17 = (_QWORD *)(v20 + 16LL * v18);
                v21 = *v17;
                if ( v24 == *v17 )
                  goto LABEL_16;
                ++v22;
              }
              if ( v23 )
                v17 = v23;
            }
LABEL_16:
            *v17 = v24;
            v17[1] = j[1];
            *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
          }
        }
        return (__int64 *)sub_C7D6A0(v3, v10 * 8, 8);
      }
      if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
      {
        v30 = 1024;
        v6 = 64;
        v27 = &v45;
        goto LABEL_45;
      }
      v29 = *(_QWORD *)(a1 + 24);
      v45 = *(_QWORD *)(a1 + 16);
      v27 = (__int64 *)v47;
      v6 = 64;
      v46 = v29;
    }
    v30 = 16LL * v6;
LABEL_45:
    *(_BYTE *)(a1 + 8) &= ~1u;
    v31 = sub_C7D670(v30, 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v31;
    goto LABEL_46;
  }
  if ( !v4 )
  {
    v7 = *(_DWORD *)(a1 + 24);
    *(_BYTE *)(a1 + 8) |= 1u;
    goto LABEL_6;
  }
  if ( v3 == (__int64 *)-4096LL || v3 == (__int64 *)-8192LL )
  {
    v27 = &v45;
  }
  else
  {
    v44 = *(_QWORD *)(a1 + 24);
    v45 = *(_QWORD *)(a1 + 16);
    v27 = (__int64 *)v47;
    v46 = v44;
  }
LABEL_46:
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v32 = *(_QWORD **)(a1 + 16);
    v33 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v32 = (_QWORD *)(a1 + 16);
    v33 = 2;
  }
  for ( k = &v32[v33]; k != v32; v32 += 2 )
  {
    if ( v32 )
      *v32 = -4096;
  }
  for ( result = &v45; result != v27; result += 2 )
  {
    v42 = *result;
    if ( *result != -4096 && v42 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v35 = (_QWORD *)(a1 + 16);
        v36 = 0;
        v37 = 0;
        v38 = a1 + 16;
      }
      else
      {
        v43 = *(_DWORD *)(a1 + 24);
        v38 = *(_QWORD *)(a1 + 16);
        if ( !v43 )
        {
          MEMORY[0] = *result;
          BUG();
        }
        v37 = v43 - 1;
        v36 = v37 & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v35 = (_QWORD *)(v38 + 16LL * v36);
      }
      v39 = *v35;
      v40 = 1;
      v41 = 0;
      if ( v42 != *v35 )
      {
        while ( v39 != -4096 )
        {
          if ( !v41 && v39 == -8192 )
            v41 = v35;
          v36 = v37 & (v40 + v36);
          v35 = (_QWORD *)(v38 + 16LL * v36);
          v39 = *v35;
          if ( v42 == *v35 )
            goto LABEL_56;
          ++v40;
        }
        if ( v41 )
          v35 = v41;
      }
LABEL_56:
      *v35 = v42;
      v35[1] = result[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return result;
}
