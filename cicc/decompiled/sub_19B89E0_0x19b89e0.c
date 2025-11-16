// Function: sub_19B89E0
// Address: 0x19b89e0
//
__int64 __fastcall sub_19B89E0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r13
  unsigned __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // r12
  __int64 *v9; // rax
  _BYTE *v10; // r13
  __int64 v11; // rdx
  _QWORD *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  _QWORD *v17; // r12
  __int64 v18; // r9
  int v19; // r8d
  int v20; // r12d
  _QWORD *v21; // r11
  unsigned int v22; // esi
  _QWORD *v23; // rdi
  __int64 v24; // r10
  __int64 v25; // rdx
  int v26; // esi
  __int64 v27; // r14
  bool v28; // zf
  __int64 *v29; // rsi
  _QWORD *v30; // rax
  __int64 v31; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v34; // r10
  int v35; // r9d
  int v36; // r14d
  _QWORD *v37; // r12
  unsigned int v38; // ecx
  _QWORD *v39; // rdi
  __int64 v40; // r11
  int v41; // ecx
  __int64 v42; // rax
  _BYTE v43[176]; // [rsp+10h] [rbp-B0h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 0xF )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v27 = *(unsigned int *)(a1 + 24);
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
      v27 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 64;
        v7 = 64;
LABEL_5:
        v9 = (__int64 *)(a1 + 16);
        v10 = v43;
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
        while ( v9 != (__int64 *)(a1 + 144) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        v12 = (_QWORD *)sub_22077B0(v8 * 8);
        v13 = *(_QWORD *)(a1 + 8);
        v14 = a1 + 16;
        *(_QWORD *)(a1 + 16) = v12;
        v15 = v13 & 1;
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 8) = v15;
        if ( (_BYTE)v15 )
        {
          v12 = (_QWORD *)(a1 + 16);
          v8 = 16;
        }
        v16 = v12;
        v17 = &v12[v8];
        while ( 1 )
        {
          if ( v16 )
            *v12 = -8;
          if ( v17 == ++v12 )
            break;
          v16 = v12;
        }
        result = (__int64)v43;
        if ( v10 != v43 )
        {
          while ( 1 )
          {
            v25 = *(_QWORD *)result;
            if ( *(_QWORD *)result != -8 && v25 != -16 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v18 = v14;
                v19 = 15;
              }
              else
              {
                v26 = *(_DWORD *)(a1 + 24);
                v18 = *(_QWORD *)(a1 + 16);
                if ( !v26 )
                  goto LABEL_74;
                v19 = v26 - 1;
              }
              v20 = 1;
              v21 = 0;
              v22 = v19 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
              v23 = (_QWORD *)(v18 + 8LL * v22);
              v24 = *v23;
              if ( v25 != *v23 )
              {
                while ( v24 != -8 )
                {
                  if ( v24 == -16 && !v21 )
                    v21 = v23;
                  v22 = v19 & (v20 + v22);
                  v23 = (_QWORD *)(v18 + 8LL * v22);
                  v24 = *v23;
                  if ( v25 == *v23 )
                    goto LABEL_23;
                  ++v20;
                }
                if ( v21 )
                  v23 = v21;
              }
LABEL_23:
              *v23 = v25;
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 8;
            if ( v10 == (_BYTE *)result )
              return result;
          }
        }
        return result;
      }
      v27 = *(unsigned int *)(a1 + 24);
      v8 = 64;
      v7 = 64;
    }
    v42 = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v42;
  }
  v28 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v29 = &v5[v27];
  if ( v28 )
  {
    v30 = *(_QWORD **)(a1 + 16);
    v31 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v30 = (_QWORD *)(a1 + 16);
    v31 = 16;
  }
  for ( i = &v30[v31]; i != v30; ++v30 )
  {
    if ( v30 )
      *v30 = -8;
  }
  for ( j = v5; v29 != j; ++j )
  {
    v25 = *j;
    if ( *j != -16 && v25 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v34 = a1 + 16;
        v35 = 15;
      }
      else
      {
        v41 = *(_DWORD *)(a1 + 24);
        v34 = *(_QWORD *)(a1 + 16);
        if ( !v41 )
        {
LABEL_74:
          MEMORY[0] = v25;
          BUG();
        }
        v35 = v41 - 1;
      }
      v36 = 1;
      v37 = 0;
      v38 = v35 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v39 = (_QWORD *)(v34 + 8LL * v38);
      v40 = *v39;
      if ( *v39 != v25 )
      {
        while ( v40 != -8 )
        {
          if ( !v37 && v40 == -16 )
            v37 = v39;
          v38 = v35 & (v36 + v38);
          v39 = (_QWORD *)(v34 + 8LL * v38);
          v40 = *v39;
          if ( v25 == *v39 )
            goto LABEL_43;
          ++v36;
        }
        if ( v37 )
          v39 = v37;
      }
LABEL_43:
      *v39 = v25;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v5);
}
