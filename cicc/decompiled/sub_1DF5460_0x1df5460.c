// Function: sub_1DF5460
// Address: 0x1df5460
//
__int64 __fastcall sub_1DF5460(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r15
  __int64 *v9; // rax
  _BYTE *v10; // r12
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // rdx
  _QWORD *v16; // rdi
  __int64 v17; // r8
  int v18; // edi
  int v19; // r11d
  _QWORD *v20; // r10
  unsigned int v21; // ecx
  _QWORD *v22; // rsi
  __int64 v23; // r9
  __int64 v24; // rdx
  int v25; // ecx
  __int64 v26; // r13
  bool v27; // zf
  __int64 *v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v33; // r10
  int v34; // r9d
  int v35; // r14d
  _QWORD *v36; // r13
  unsigned int v37; // ecx
  _QWORD *v38; // rdi
  __int64 v39; // r11
  int v40; // ecx
  __int64 v41; // rax
  _BYTE v42[112]; // [rsp+10h] [rbp-70h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 7 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v26 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  else
  {
    v5 = *(__int64 **)(a1 + 16);
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
      v8 = (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v26 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 64;
        v7 = 64;
LABEL_5:
        v9 = (__int64 *)(a1 + 16);
        v10 = v42;
        do
        {
          v11 = *v9;
          if ( *v9 != -8 && v11 != -16 )
          {
            if ( v10 )
              *(_QWORD *)v10 = v11;
            v10 += 8;
          }
          ++v9;
        }
        while ( v9 != (__int64 *)(a1 + 80) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v12 = (_QWORD *)sub_22077B0(v8 * 8);
        v13 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = v12;
        v14 = v13 & 1;
        *(_QWORD *)(a1 + 8) = v14;
        if ( (_BYTE)v14 )
        {
          v12 = (_QWORD *)(a1 + 16);
          v8 = 8;
        }
        v15 = v12;
        v16 = &v12[v8];
        while ( 1 )
        {
          if ( v15 )
            *v12 = -8;
          if ( v16 == ++v12 )
            break;
          v15 = v12;
        }
        result = (__int64)v42;
        if ( v10 != v42 )
        {
          while ( 1 )
          {
            v24 = *(_QWORD *)result;
            if ( *(_QWORD *)result != -8 && v24 != -16 )
            {
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
                  goto LABEL_74;
                v18 = v25 - 1;
              }
              v19 = 1;
              v20 = 0;
              v21 = v18 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
              v22 = (_QWORD *)(v17 + 8LL * v21);
              v23 = *v22;
              if ( v24 != *v22 )
              {
                while ( v23 != -8 )
                {
                  if ( v23 == -16 && !v20 )
                    v20 = v22;
                  v21 = v18 & (v19 + v21);
                  v22 = (_QWORD *)(v17 + 8LL * v21);
                  v23 = *v22;
                  if ( v24 == *v22 )
                    goto LABEL_23;
                  ++v19;
                }
                if ( v20 )
                  v22 = v20;
              }
LABEL_23:
              *v22 = v24;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 8;
            if ( v10 == (_BYTE *)result )
              return result;
          }
        }
        return result;
      }
      v26 = *(unsigned int *)(a1 + 24);
      v8 = 64;
      v7 = 64;
    }
    v41 = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v41;
  }
  v27 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v28 = &v5[v26];
  if ( v27 )
  {
    v29 = *(_QWORD **)(a1 + 16);
    v30 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v29 = (_QWORD *)(a1 + 16);
    v30 = 8;
  }
  for ( i = &v29[v30]; i != v29; ++v29 )
  {
    if ( v29 )
      *v29 = -8;
  }
  for ( j = v5; v28 != j; ++j )
  {
    v24 = *j;
    if ( *j != -16 && v24 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v33 = a1 + 16;
        v34 = 7;
      }
      else
      {
        v40 = *(_DWORD *)(a1 + 24);
        v33 = *(_QWORD *)(a1 + 16);
        if ( !v40 )
        {
LABEL_74:
          MEMORY[0] = v24;
          BUG();
        }
        v34 = v40 - 1;
      }
      v35 = 1;
      v36 = 0;
      v37 = v34 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v38 = (_QWORD *)(v33 + 8LL * v37);
      v39 = *v38;
      if ( *v38 != v24 )
      {
        while ( v39 != -8 )
        {
          if ( !v36 && v39 == -16 )
            v36 = v38;
          v37 = v34 & (v35 + v37);
          v38 = (_QWORD *)(v33 + 8LL * v37);
          v39 = *v38;
          if ( v24 == *v38 )
            goto LABEL_43;
          ++v35;
        }
        if ( v36 )
          v38 = v36;
      }
LABEL_43:
      *v38 = v24;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v5);
}
