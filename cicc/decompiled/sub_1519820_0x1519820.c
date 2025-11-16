// Function: sub_1519820
// Address: 0x1519820
//
__int64 __fastcall sub_1519820(__int64 a1, int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r13
  unsigned __int64 v6; // rax
  int v7; // r15d
  __int64 v8; // r12
  __int64 v9; // rax
  char *v10; // r14
  _QWORD *v11; // rax
  bool v12; // zf
  _QWORD *v13; // rdi
  _QWORD *v14; // rcx
  unsigned int v15; // esi
  int v16; // edi
  __int64 v17; // r8
  __int64 v18; // r9
  int v19; // r11d
  _QWORD *v20; // r10
  __int64 v21; // rdx
  int v22; // edi
  __int64 v23; // r14
  __int64 *v24; // rsi
  _QWORD *v25; // rax
  __int64 v26; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  _QWORD *v29; // rcx
  unsigned int v30; // edi
  int v31; // r8d
  __int64 v32; // r9
  __int64 v33; // r10
  int v34; // r14d
  _QWORD *v35; // r11
  int v36; // r8d
  __int64 v37; // rax
  _QWORD v38[2]; // [rsp+0h] [rbp-40h] BYREF
  char v39; // [rsp+10h] [rbp-30h] BYREF

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
      v8 = 2LL * (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v23 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 128;
        v7 = 64;
LABEL_5:
        if ( v5 == (__int64 *)-8LL || v5 == (__int64 *)-16LL )
        {
          v10 = (char *)v38;
        }
        else
        {
          v9 = *(_QWORD *)(a1 + 24);
          v38[0] = *(_QWORD *)(a1 + 16);
          v10 = &v39;
          v38[1] = v9;
        }
        *(_BYTE *)(a1 + 8) &= ~1u;
        v11 = (_QWORD *)sub_22077B0(v8 * 8);
        v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
        *(_QWORD *)(a1 + 8) &= 1uLL;
        *(_QWORD *)(a1 + 16) = v11;
        *(_DWORD *)(a1 + 24) = v7;
        if ( !v12 )
        {
          v11 = (_QWORD *)(a1 + 16);
          v8 = 2;
        }
        v13 = &v11[v8];
        do
        {
          if ( v11 )
            *v11 = -8;
          v11 += 2;
        }
        while ( v13 != v11 );
        result = (__int64)v38;
        if ( v10 != (char *)v38 )
        {
          while ( 1 )
          {
            v21 = *(_QWORD *)result;
            if ( *(_QWORD *)result != -16 && v21 != -8 )
            {
              if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
              {
                v14 = (_QWORD *)(a1 + 16);
                v15 = 0;
                v16 = 0;
                v17 = a1 + 16;
              }
              else
              {
                v22 = *(_DWORD *)(a1 + 24);
                v17 = *(_QWORD *)(a1 + 16);
                if ( !v22 )
                  goto LABEL_70;
                v16 = v22 - 1;
                v15 = v16 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
                v14 = (_QWORD *)(v17 + 16LL * v15);
              }
              v18 = *v14;
              v19 = 1;
              v20 = 0;
              if ( v21 != *v14 )
              {
                while ( v18 != -8 )
                {
                  if ( !v20 && v18 == -16 )
                    v20 = v14;
                  v15 = v16 & (v19 + v15);
                  v14 = (_QWORD *)(v17 + 16LL * v15);
                  v18 = *v14;
                  if ( v21 == *v14 )
                    goto LABEL_18;
                  ++v19;
                }
                if ( v20 )
                  v14 = v20;
              }
LABEL_18:
              *v14 = v21;
              v14[1] = *(_QWORD *)(result + 8);
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
            }
            result += 16;
            if ( v10 == (char *)result )
              return result;
          }
        }
        return result;
      }
      v23 = *(unsigned int *)(a1 + 24);
      v8 = 128;
      v7 = 64;
    }
    v37 = sub_22077B0(v8 * 8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v37;
  }
  else
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v23 = *(unsigned int *)(a1 + 24);
    *(_BYTE *)(a1 + 8) = result | 1;
  }
  v12 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v24 = &v5[2 * v23];
  if ( v12 )
  {
    v25 = *(_QWORD **)(a1 + 16);
    v26 = 2LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v25 = (_QWORD *)(a1 + 16);
    v26 = 2;
  }
  for ( i = &v25[v26]; i != v25; v25 += 2 )
  {
    if ( v25 )
      *v25 = -8;
  }
  for ( j = v5; v24 != j; j += 2 )
  {
    v21 = *j;
    if ( *j != -16 && v21 != -8 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v29 = (_QWORD *)(a1 + 16);
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
LABEL_70:
          MEMORY[0] = v21;
          BUG();
        }
        v31 = v36 - 1;
        v30 = v31 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v29 = (_QWORD *)(v32 + 16LL * v30);
      }
      v33 = *v29;
      v34 = 1;
      v35 = 0;
      if ( *v29 != v21 )
      {
        while ( v33 != -8 )
        {
          if ( !v35 && v33 == -16 )
            v35 = v29;
          v30 = v31 & (v34 + v30);
          v29 = (_QWORD *)(v32 + 16LL * v30);
          v33 = *v29;
          if ( v21 == *v29 )
            goto LABEL_38;
          ++v34;
        }
        if ( v35 )
          v29 = v35;
      }
LABEL_38:
      *v29 = v21;
      v29[1] = j[1];
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v5);
}
