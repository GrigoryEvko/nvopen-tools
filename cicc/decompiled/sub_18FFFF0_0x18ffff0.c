// Function: sub_18FFFF0
// Address: 0x18ffff0
//
__int64 __fastcall sub_18FFFF0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  char v4; // dl
  __int64 *v5; // r12
  unsigned __int64 v6; // rax
  int v7; // r14d
  __int64 v8; // r15
  __int64 *v9; // rax
  __int64 *v10; // r12
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 *v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdi
  __int64 v17; // r13
  bool v18; // zf
  __int64 *v19; // rsi
  _QWORD *v20; // rax
  __int64 v21; // rdx
  _QWORD *i; // rdx
  __int64 *j; // rax
  __int64 v24; // r10
  int v25; // r9d
  int v26; // r14d
  _QWORD *v27; // r13
  unsigned int v28; // ecx
  _QWORD *v29; // rdi
  __int64 v30; // r11
  __int64 v31; // rdx
  int v32; // ecx
  __int64 v33; // rax
  __int64 v34; // r8
  int v35; // edi
  int v36; // r11d
  __int64 *v37; // r10
  unsigned int v38; // edx
  __int64 *v39; // rsi
  __int64 v40; // r9
  int v41; // edx
  _BYTE v42[80]; // [rsp+10h] [rbp-50h] BYREF

  result = *(unsigned __int8 *)(a1 + 8);
  v4 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 3 )
  {
    if ( v4 )
      return result;
    v5 = *(__int64 **)(a1 + 16);
    v17 = *(unsigned int *)(a1 + 24);
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
      v8 = 8LL * (unsigned int)v6;
      if ( v4 )
        goto LABEL_5;
      v17 = *(unsigned int *)(a1 + 24);
    }
    else
    {
      if ( v4 )
      {
        v8 = 512;
        v7 = 64;
LABEL_5:
        v9 = (__int64 *)(a1 + 16);
        v10 = (__int64 *)v42;
        do
        {
          v11 = *v9;
          if ( *v9 != -8 && v11 != -16 )
          {
            if ( v10 )
              *v10 = v11;
            ++v10;
          }
          ++v9;
        }
        while ( v9 != (__int64 *)(a1 + 48) );
        *(_BYTE *)(a1 + 8) &= ~1u;
        result = sub_22077B0(v8);
        v12 = *(_QWORD *)(a1 + 8);
        *(_DWORD *)(a1 + 24) = v7;
        *(_QWORD *)(a1 + 16) = result;
        v13 = (__int64 *)v42;
        v14 = v12 & 1;
        *(_QWORD *)(a1 + 8) = v14;
        if ( (_BYTE)v14 )
        {
          result = a1 + 16;
          v8 = 32;
        }
        v15 = result;
        v16 = result + v8;
        while ( 1 )
        {
          if ( v15 )
            *(_QWORD *)result = -8;
          result += 8;
          if ( v16 == result )
            break;
          v15 = result;
        }
        while ( v10 != v13 )
        {
          result = *v13;
          if ( *v13 != -8 && result != -16 )
          {
            if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
            {
              v34 = a1 + 16;
              v35 = 3;
            }
            else
            {
              v41 = *(_DWORD *)(a1 + 24);
              v34 = *(_QWORD *)(a1 + 16);
              if ( !v41 )
              {
                MEMORY[0] = *v13;
                BUG();
              }
              v35 = v41 - 1;
            }
            v36 = 1;
            v37 = 0;
            v38 = v35 & (((unsigned int)result >> 9) ^ ((unsigned int)result >> 4));
            v39 = (__int64 *)(v34 + 8LL * v38);
            v40 = *v39;
            if ( result != *v39 )
            {
              while ( v40 != -8 )
              {
                if ( v40 == -16 && !v37 )
                  v37 = v39;
                v38 = v35 & (v36 + v38);
                v39 = (__int64 *)(v34 + 8LL * v38);
                v40 = *v39;
                if ( result == *v39 )
                  goto LABEL_48;
                ++v36;
              }
              if ( v37 )
                v39 = v37;
            }
LABEL_48:
            *v39 = result;
            result = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1u;
            *(_DWORD *)(a1 + 8) = result;
          }
          ++v13;
        }
        return result;
      }
      v17 = *(unsigned int *)(a1 + 24);
      v8 = 512;
      v7 = 64;
    }
    v33 = sub_22077B0(v8);
    *(_DWORD *)(a1 + 24) = v7;
    *(_QWORD *)(a1 + 16) = v33;
  }
  v18 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v19 = &v5[v17];
  if ( v18 )
  {
    v20 = *(_QWORD **)(a1 + 16);
    v21 = *(unsigned int *)(a1 + 24);
  }
  else
  {
    v20 = (_QWORD *)(a1 + 16);
    v21 = 4;
  }
  for ( i = &v20[v21]; i != v20; ++v20 )
  {
    if ( v20 )
      *v20 = -8;
  }
  for ( j = v5; v19 != j; ++j )
  {
    v31 = *j;
    if ( *j != -16 && v31 != -8 )
    {
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
          MEMORY[0] = *j;
          BUG();
        }
        v25 = v32 - 1;
      }
      v26 = 1;
      v27 = 0;
      v28 = v25 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
      v29 = (_QWORD *)(v24 + 8LL * v28);
      v30 = *v29;
      if ( *v29 != v31 )
      {
        while ( v30 != -8 )
        {
          if ( !v27 && v30 == -16 )
            v27 = v29;
          v28 = v25 & (v26 + v28);
          v29 = (_QWORD *)(v24 + 8LL * v28);
          v30 = *v29;
          if ( v31 == *v29 )
            goto LABEL_33;
          ++v26;
        }
        if ( v27 )
          v29 = v27;
      }
LABEL_33:
      *v29 = v31;
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    }
  }
  return j___libc_free_0(v5);
}
