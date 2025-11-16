// Function: sub_2D9FC50
// Address: 0x2d9fc50
//
_QWORD *__fastcall sub_2D9FC50(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  unsigned int v4; // ebx
  __int64 v5; // r14
  unsigned int v6; // edi
  _QWORD *result; // rax
  __int64 *v8; // r13
  _QWORD *i; // rdx
  __int64 *j; // rbx
  __int64 v11; // rcx
  int v12; // edi
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 *v15; // r11
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned int k; // eax
  __int64 *v19; // rdi
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // rax
  volatile signed __int32 *v23; // r15
  signed __int32 v24; // eax
  signed __int32 v25; // eax
  int v26; // eax
  _QWORD *m; // rdx
  __int64 v28; // [rsp+8h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(_DWORD *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v6 = (((((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
        | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
        | ((v2 | (v2 >> 1)) >> 2)
        | v2
        | (v2 >> 1)) >> 16)
      | ((((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4) | ((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 8)
      | ((((v2 | (v2 >> 1)) >> 2) | v2 | (v2 >> 1)) >> 4)
      | ((v2 | (v2 >> 1)) >> 2)
      | v2
      | (v2 >> 1))
     + 1;
  if ( v6 < 0x40 )
    v6 = 64;
  *(_DWORD *)(a1 + 24) = v6;
  result = (_QWORD *)sub_C7D670(32LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v28 = 32LL * v4;
    v8 = (__int64 *)(v5 + v28);
    for ( i = &result[4 * *(unsigned int *)(a1 + 24)]; i != result; result += 4 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
    if ( v8 != (__int64 *)v5 )
    {
      for ( j = (__int64 *)v5; v8 != j; j += 4 )
      {
        while ( 1 )
        {
          v11 = *j;
          if ( *j != -4096 )
            break;
          if ( j[1] == -4096 )
          {
LABEL_22:
            j += 4;
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
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
            v14 = *(_QWORD *)(a1 + 8);
            v15 = 0;
            v16 = (unsigned int)(v12 - 1);
            v17 = 1;
            for ( k = v16
                    & (((0xBF58476D1CE4E5B9LL
                       * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)
                        | ((unsigned __int64)(((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4)) << 32))) >> 31)
                     ^ (484763065 * (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4)))); ; k = v16 & v26 )
            {
              v19 = (__int64 *)(v14 + 32LL * k);
              v20 = *v19;
              if ( v11 == *v19 && v19[1] == v13 )
                break;
              if ( v20 == -4096 )
              {
                if ( v19[1] == -4096 )
                {
                  if ( v15 )
                    v19 = v15;
                  break;
                }
              }
              else if ( v20 == -8192 && v19[1] == -8192 && !v15 )
              {
                v15 = (__int64 *)(v14 + 32LL * k);
              }
              v26 = v17 + k;
              v17 = (unsigned int)(v17 + 1);
            }
            *v19 = v11;
            v19[1] = j[1];
            v21 = j[2];
            v19[3] = 0;
            v19[2] = v21;
            v22 = j[3];
            j[3] = 0;
            v19[3] = v22;
            j[2] = 0;
            ++*(_DWORD *)(a1 + 16);
            v23 = (volatile signed __int32 *)j[3];
            if ( !v23 )
              goto LABEL_22;
            if ( &_pthread_key_create )
            {
              v24 = _InterlockedExchangeAdd(v23 + 2, 0xFFFFFFFF);
            }
            else
            {
              v24 = *((_DWORD *)v23 + 2);
              v13 = (unsigned int)(v24 - 1);
              *((_DWORD *)v23 + 2) = v13;
            }
            if ( v24 != 1 )
              goto LABEL_22;
            (*(void (__fastcall **)(volatile signed __int32 *, __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v23 + 16LL))(
              v23,
              v14,
              v13,
              v11,
              v16,
              v17);
            if ( &_pthread_key_create )
            {
              v25 = _InterlockedExchangeAdd(v23 + 3, 0xFFFFFFFF);
            }
            else
            {
              v25 = *((_DWORD *)v23 + 3);
              *((_DWORD *)v23 + 3) = v25 - 1;
            }
            if ( v25 != 1 )
              goto LABEL_22;
            j += 4;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v23 + 24LL))(v23);
            if ( v8 == j )
              return (_QWORD *)sub_C7D6A0(v5, v28, 8);
          }
        }
        if ( v11 != -8192 || j[1] != -8192 )
          goto LABEL_12;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    for ( m = &result[4 * *(unsigned int *)(a1 + 24)]; m != result; result += 4 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
      }
    }
  }
  return result;
}
