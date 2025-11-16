// Function: sub_1519BD0
// Address: 0x1519bd0
//
__int64 __fastcall sub_1519BD0(__int64 a1, int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r13
  unsigned __int64 v6; // rax
  int v7; // esi
  __int64 v8; // r15
  __int64 v9; // rax
  char *v10; // r13
  bool v11; // zf
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 *i; // rcx
  int v15; // edi
  __int64 v16; // r8
  int v17; // edi
  unsigned int v18; // edx
  _QWORD *v19; // rax
  __int64 v20; // r9
  int v21; // r14d
  _QWORD *v22; // r10
  __int64 v23; // rdx
  __int64 v24; // rsi
  __int64 v25; // r14
  __int64 *v26; // r14
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *j; // rdx
  __int64 *k; // rbx
  _QWORD *v31; // rdx
  __int64 v32; // r8
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rdi
  int v36; // r11d
  _QWORD *v37; // r9
  __int64 v38; // rdx
  __int64 v39; // rdi
  __int64 v40; // rax
  int v41; // ecx
  __int64 v42; // rdx
  _QWORD v43[2]; // [rsp+10h] [rbp-40h] BYREF
  char v44; // [rsp+20h] [rbp-30h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 )
  {
    v5 = *(__int64 **)(a1 + 16);
    v6 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
    v7 = v6;
    if ( (unsigned int)v6 > 0x40 )
    {
      v8 = 16LL * (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v25 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 1024;
        v7 = 64;
LABEL_5:
        if ( v5 == (__int64 *)-8LL || v5 == (__int64 *)-16LL )
        {
          v10 = (char *)v43;
        }
        else
        {
          v9 = *(_QWORD *)(a1 + 24);
          v43[0] = *(_QWORD *)(a1 + 16);
          v10 = &v44;
          v43[1] = v9;
        }
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v8);
        v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = result;
        v12 = result;
        *(_DWORD *)(a1 + 24) = v7;
        if ( !v11 )
        {
          v12 = a1 + 16;
          result = a1 + 16;
          v8 = 16;
        }
        v13 = result + v8;
        while ( 1 )
        {
          if ( v12 )
            *(_QWORD *)result = -8;
          result += 16;
          if ( v13 == result )
            break;
          v12 = result;
        }
        for ( i = v43; i != (__int64 *)v10; i += 2 )
        {
          v24 = *i;
          if ( *i != -8 && v24 != -16 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v19 = (_QWORD *)(a1 + 16);
              v16 = a1 + 16;
              v18 = 0;
              v17 = 0;
            }
            else
            {
              v15 = *(_DWORD *)(a1 + 24);
              v16 = *(_QWORD *)(a1 + 16);
              if ( !v15 )
              {
                MEMORY[0] = *i;
                BUG();
              }
              v17 = v15 - 1;
              v18 = v17 & (((unsigned int)v24 >> 4) ^ ((unsigned int)v24 >> 9));
              v19 = (_QWORD *)(v16 + 16LL * v18);
            }
            v20 = *v19;
            v21 = 1;
            v22 = 0;
            if ( v24 != *v19 )
            {
              while ( v20 != -8 )
              {
                if ( v20 == -16 && !v22 )
                  v22 = v19;
                v18 = v17 & (v21 + v18);
                v19 = (_QWORD *)(v16 + 16LL * v18);
                v20 = *v19;
                if ( v24 == *v19 )
                  goto LABEL_20;
                ++v21;
              }
              if ( v22 )
                v19 = v22;
            }
LABEL_20:
            v23 = i[1];
            *v19 = v24;
            v19[1] = v23;
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
        }
        return result;
      }
      v25 = *(unsigned int *)(a1 + 24);
      v8 = 1024;
      v7 = 64;
    }
    *(_QWORD *)(a1 + 16) = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
  }
  else
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v25 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  v26 = &v5[2 * v25];
  v11 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  if ( v11 )
  {
    v27 = *(_QWORD **)(a1 + 16);
    v28 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v27 = (_QWORD *)(a1 + 16);
    v28 = 2;
  }
  for ( j = &v27[v28]; j != v27; v27 += 2 )
  {
    if ( v27 )
      *v27 = -8;
  }
  for ( k = v5; v26 != k; k += 2 )
  {
    v40 = *k;
    if ( *k != -8 && v40 != -16 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v31 = (_QWORD *)(a1 + 16);
        v32 = 0;
        v33 = 0;
        v34 = a1 + 16;
      }
      else
      {
        v41 = *(_DWORD *)(a1 + 24);
        v34 = *(_QWORD *)(a1 + 16);
        if ( !v41 )
        {
          MEMORY[0] = *k;
          BUG();
        }
        v33 = (unsigned int)(v41 - 1);
        v32 = (unsigned int)v33 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v31 = (_QWORD *)(v34 + 16 * v32);
      }
      v35 = *v31;
      v36 = 1;
      v37 = 0;
      if ( *v31 != v40 )
      {
        while ( v35 != -8 )
        {
          if ( v35 == -16 && !v37 )
            v37 = v31;
          v42 = (unsigned int)v33 & ((_DWORD)v32 + v36);
          v32 = v42;
          v31 = (_QWORD *)(v34 + 16 * v42);
          v35 = *v31;
          if ( v40 == *v31 )
            goto LABEL_39;
          ++v36;
        }
        if ( v37 )
          v31 = v37;
      }
LABEL_39:
      *v31 = v40;
      v31[1] = k[1];
      k[1] = 0;
      v38 = (unsigned int)(2 * (*(_DWORD *)(a1 + 8) >> 1) + 2);
      *(_DWORD *)(a1 + 8) = v38 | *(_DWORD *)(a1 + 8) & 1;
      v39 = k[1];
      if ( v39 )
        sub_16307F0(v39, v34, v38, v33, v32);
    }
  }
  return j___libc_free_0(v5);
}
