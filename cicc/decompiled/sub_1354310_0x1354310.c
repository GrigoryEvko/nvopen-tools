// Function: sub_1354310
// Address: 0x1354310
//
__int64 __fastcall sub_1354310(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  unsigned __int64 v5; // rax
  int v6; // r14d
  __int64 v7; // rdi
  __int64 *v8; // rax
  _BYTE *v9; // r12
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rdx
  _QWORD *v14; // rdx
  _QWORD *v15; // rdi
  __int64 v16; // r8
  int v17; // edi
  int v18; // r11d
  _QWORD *v19; // r10
  unsigned int v20; // ecx
  _QWORD *v21; // rsi
  __int64 v22; // r9
  unsigned __int64 v23; // rdx
  int v24; // ecx
  unsigned __int64 *v25; // r12
  __int64 v26; // r13
  bool v27; // zf
  unsigned __int64 *v28; // rsi
  _QWORD *v29; // rax
  __int64 v30; // rdx
  _QWORD *i; // rdx
  unsigned __int64 *j; // rax
  __int64 v33; // r10
  int v34; // r9d
  int v35; // r14d
  unsigned __int64 *v36; // r13
  unsigned int v37; // ecx
  unsigned __int64 *v38; // rdi
  unsigned __int64 v39; // r11
  int v40; // ecx
  __int64 v41; // rax
  _BYTE v42[176]; // [rsp+10h] [rbp-B0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0xF )
  {
    if ( v4 )
      return result;
    v25 = *(unsigned __int64 **)(a1 + 16);
    v26 = *(unsigned int *)(a1 + 24);
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
      v7 = (unsigned int)v5;
      if ( v4 )
        goto LABEL_5;
      v25 = *(unsigned __int64 **)(a1 + 16);
      v26 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v7 = 64;
        v6 = 64;
LABEL_5:
        v8 = (__int64 *)(a1 + 16);
        v9 = v42;
        do
        {
          v10 = *v8;
          if ( *v8 != -4 && v10 != -16 )
          {
            if ( v9 )
              *(_QWORD *)v9 = v10;
            v9 += 8;
          }
          ++v8;
        }
        while ( v8 != (__int64 *)(a1 + 144) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v11 = (_QWORD *)sub_22077B0(v7 * 8);
        v12 = *(_QWORD *)(a1 + 8);
        *(_QWORD *)(a1 + 16) = v11;
        v13 = v12 & 1;
        *(_DWORD *)(a1 + 24) = v6;
        *(_QWORD *)(a1 + 8) = v13;
        if ( (_BYTE)v13 )
        {
          v11 = (_QWORD *)(a1 + 16);
          v7 = 16;
        }
        v14 = v11;
        v15 = &v11[v7];
        while ( 1 )
        {
          if ( v14 )
            *v11 = -4;
          if ( v15 == ++v11 )
            break;
          v14 = v11;
        }
        result = (__int64)v42;
        if ( v9 != v42 )
        {
          while ( 1 )
          {
            v23 = *(_QWORD *)result;
            if ( *(_QWORD *)result != -4 && v23 != -16 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v16 = a1 + 16;
                v17 = 15;
              }
              else
              {
                v24 = *(_DWORD *)(a1 + 24);
                v16 = *(_QWORD *)(a1 + 16);
                if ( !v24 )
                  goto LABEL_74;
                v17 = v24 - 1;
              }
              v18 = 1;
              v19 = 0;
              v20 = v17 & (v23 ^ (v23 >> 9));
              v21 = (_QWORD *)(v16 + 8LL * v20);
              v22 = *v21;
              if ( v23 != *v21 )
              {
                while ( v22 != -4 )
                {
                  if ( v22 == -16 && !v19 )
                    v19 = v21;
                  v20 = v17 & (v18 + v20);
                  v21 = (_QWORD *)(v16 + 8LL * v20);
                  v22 = *v21;
                  if ( v23 == *v21 )
                    goto LABEL_23;
                  ++v18;
                }
                if ( v19 )
                  v21 = v19;
              }
LABEL_23:
              *v21 = *(_QWORD *)result;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 8;
            if ( v9 == (_BYTE *)result )
              return result;
          }
        }
        return result;
      }
      v25 = *(unsigned __int64 **)(a1 + 16);
      v26 = *(unsigned int *)(a1 + 24);
      v6 = 64;
      v7 = 64;
    }
    v41 = sub_22077B0(v7 * 8);
    *(_DWORD *)(a1 + 24) = v6;
    *(_QWORD *)(a1 + 16) = v41;
  }
  v27 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v28 = &v25[v26];
  if ( v27 )
  {
    v29 = *(_QWORD **)(a1 + 16);
    v30 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v29 = (_QWORD *)(a1 + 16);
    v30 = 16;
  }
  for ( i = &v29[v30]; i != v29; ++v29 )
  {
    if ( v29 )
      *v29 = -4;
  }
  for ( j = v25; v28 != j; ++j )
  {
    v23 = *j;
    if ( *j != -16 && v23 != -4 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v33 = a1 + 16;
        v34 = 15;
      }
      else
      {
        v40 = *(_DWORD *)(a1 + 24);
        v33 = *(_QWORD *)(a1 + 16);
        if ( !v40 )
        {
LABEL_74:
          MEMORY[0] = v23;
          BUG();
        }
        v34 = v40 - 1;
      }
      v35 = 1;
      v36 = 0;
      v37 = v34 & (v23 ^ (v23 >> 9));
      v38 = (unsigned __int64 *)(v33 + 8LL * v37);
      v39 = *v38;
      if ( *v38 != v23 )
      {
        while ( v39 != -4 )
        {
          if ( !v36 && v39 == -16 )
            v36 = v38;
          v37 = v34 & (v35 + v37);
          v38 = (unsigned __int64 *)(v33 + 8LL * v37);
          v39 = *v38;
          if ( v23 == *v38 )
            goto LABEL_43;
          ++v35;
        }
        if ( v36 )
          v38 = v36;
      }
LABEL_43:
      *v38 = *j;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v25);
}
