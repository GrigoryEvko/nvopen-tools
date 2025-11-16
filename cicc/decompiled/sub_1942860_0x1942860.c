// Function: sub_1942860
// Address: 0x1942860
//
_QWORD *__fastcall sub_1942860(__int64 a1, int a2)
{
  __int64 v3; // rbx
  __int64 *v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  __int64 *v8; // r14
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rdx
  int v12; // ecx
  __int64 v13; // rsi
  int v14; // ecx
  __int64 v15; // rdi
  __int64 *v16; // r10
  __int64 v17; // r8
  int v18; // r9d
  unsigned __int64 v19; // r8
  unsigned __int64 v20; // r8
  unsigned int k; // eax
  __int64 *v22; // r8
  __int64 v23; // r11
  int v24; // eax
  __int64 v25; // rdi
  __int64 v26; // rdi
  unsigned int v27; // eax
  __int64 v28; // rdx
  _QWORD *m; // rdx

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(__int64 **)(a1 + 8);
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
  if ( (unsigned int)v5 < 0x40 )
    LODWORD(v5) = 64;
  *(_DWORD *)(a1 + 24) = v5;
  result = (_QWORD *)sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[6 * v3];
    for ( i = &result[6 * v7]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
    if ( v8 != v4 )
    {
      for ( j = v4; v8 != j; j += 6 )
      {
        while ( 1 )
        {
          v11 = *j;
          if ( *j != -8 )
            break;
          if ( j[1] == -8 )
          {
LABEL_22:
            j += 6;
            if ( v8 == j )
              return (_QWORD *)j___libc_free_0(v4);
          }
          else
          {
LABEL_12:
            v12 = *(_DWORD *)(a1 + 24);
            if ( !v12 )
            {
              MEMORY[0] = *j;
              BUG();
            }
            v13 = j[1];
            v14 = v12 - 1;
            v16 = 0;
            v17 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
            v18 = 1;
            v19 = (((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                  - 1
                  - (v17 << 32)) >> 22)
                ^ ((v17 | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))
                 - 1
                 - (v17 << 32));
            v20 = ((9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13)))) >> 15)
                ^ (9 * (((v19 - 1 - (v19 << 13)) >> 8) ^ (v19 - 1 - (v19 << 13))));
            for ( k = v14 & (((v20 - 1 - (v20 << 27)) >> 31) ^ (v20 - 1 - ((_DWORD)v20 << 27))); ; k = v14 & v27 )
            {
              v15 = *(_QWORD *)(a1 + 8);
              v22 = (__int64 *)(v15 + 48LL * k);
              v23 = *v22;
              if ( v11 == *v22 && v22[1] == v13 )
                break;
              if ( v23 == -8 )
              {
                if ( v22[1] == -8 )
                {
                  if ( v16 )
                    v22 = v16;
                  break;
                }
              }
              else if ( v23 == -16 && v22[1] == -16 && !v16 )
              {
                v16 = (__int64 *)(v15 + 48LL * k);
              }
              v27 = v18 + k;
              ++v18;
            }
            *v22 = v11;
            v22[1] = j[1];
            *((_DWORD *)v22 + 6) = *((_DWORD *)j + 6);
            v22[2] = j[2];
            v24 = *((_DWORD *)j + 10);
            *((_DWORD *)j + 6) = 0;
            *((_DWORD *)v22 + 10) = v24;
            v22[4] = j[4];
            *((_DWORD *)j + 10) = 0;
            ++*(_DWORD *)(a1 + 16);
            if ( *((_DWORD *)j + 10) > 0x40u )
            {
              v25 = j[4];
              if ( v25 )
                j_j___libc_free_0_0(v25);
            }
            if ( *((_DWORD *)j + 6) <= 0x40u )
              goto LABEL_22;
            v26 = j[2];
            if ( !v26 )
              goto LABEL_22;
            j_j___libc_free_0_0(v26);
            j += 6;
            if ( v8 == j )
              return (_QWORD *)j___libc_free_0(v4);
          }
        }
        if ( v11 != -16 || j[1] != -16 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v28 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[6 * v28]; m != result; result += 6 )
    {
      if ( result )
      {
        *result = -8;
        result[1] = -8;
      }
    }
  }
  return result;
}
