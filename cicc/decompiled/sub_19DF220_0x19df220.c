// Function: sub_19DF220
// Address: 0x19df220
//
_QWORD *__fastcall sub_19DF220(__int64 a1, int a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r13
  _QWORD *i; // rdx
  _QWORD *v9; // r15
  __int64 v10; // rdx
  int v11; // ecx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  __int64 *v15; // r9
  unsigned int v16; // ecx
  __int64 *v17; // r12
  __int64 v18; // rdi
  unsigned __int64 *v19; // rdx
  unsigned int v20; // r14d
  _QWORD *v21; // r14
  _QWORD *v22; // r12
  __int64 v23; // rdx
  _QWORD *v24; // rcx
  __int64 v25; // rsi
  _QWORD *v26; // rcx
  _QWORD *j; // r9
  unsigned __int64 v28; // rsi
  _QWORD *v29; // r14
  _QWORD *v30; // r12
  __int64 v31; // rdx
  __int64 v32; // rdx
  _QWORD *k; // rdx
  _QWORD *v34; // [rsp+0h] [rbp-50h]
  _QWORD *v35; // [rsp+8h] [rbp-48h]
  unsigned __int64 *v36; // [rsp+10h] [rbp-40h]
  _QWORD *v37; // [rsp+18h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
  v37 = v4;
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
  result = (_QWORD *)sub_22077B0(72LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[9 * v3];
    for ( i = &result[9 * *(unsigned int *)(a1 + 24)]; i != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
    v9 = v4 + 3;
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v10 = *(v9 - 3);
        if ( v10 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *(v9 - 3);
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = (__int64 *)(v13 + 72LL * v16);
          v18 = *v17;
          if ( v10 != *v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (__int64 *)(v13 + 72LL * v16);
              v18 = *v17;
              if ( v10 == *v17 )
                goto LABEL_13;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_13:
          *v17 = v10;
          v19 = (unsigned __int64 *)(v17 + 3);
          v17[1] = (__int64)(v17 + 3);
          v17[2] = 0x200000000LL;
          v20 = *((_DWORD *)v9 - 2);
          if ( v17 + 1 != v9 - 2 && v20 )
          {
            v24 = (_QWORD *)*(v9 - 2);
            if ( v24 == v9 )
            {
              v25 = v20;
              v26 = v9;
              if ( v20 > 2 )
              {
                sub_170B450((__int64)(v17 + 1), v20);
                v19 = (unsigned __int64 *)v17[1];
                v26 = (_QWORD *)*(v9 - 2);
                v25 = *((unsigned int *)v9 - 2);
              }
              for ( j = &v26[3 * v25]; j != v26; v19 += 3 )
              {
                if ( v19 )
                {
                  *v19 = 6;
                  v19[1] = 0;
                  v28 = v26[2];
                  v19[2] = v28;
                  if ( v28 != 0 && v28 != -8 && v28 != -16 )
                  {
                    v34 = j;
                    v35 = v26;
                    v36 = v19;
                    sub_1649AC0(v19, *v26 & 0xFFFFFFFFFFFFFFF8LL);
                    j = v34;
                    v26 = v35;
                    v19 = v36;
                  }
                }
                v26 += 3;
              }
              *((_DWORD *)v17 + 4) = v20;
              v29 = (_QWORD *)*(v9 - 2);
              v30 = &v29[3 * *((unsigned int *)v9 - 2)];
              while ( v29 != v30 )
              {
                v31 = *(v30 - 1);
                v30 -= 3;
                if ( v31 != 0 && v31 != -8 && v31 != -16 )
                  sub_1649B30(v30);
              }
              *((_DWORD *)v9 - 2) = 0;
            }
            else
            {
              v17[1] = (__int64)v24;
              *((_DWORD *)v17 + 4) = *((_DWORD *)v9 - 2);
              *((_DWORD *)v17 + 5) = *((_DWORD *)v9 - 1);
              *(v9 - 2) = v9;
              *((_DWORD *)v9 - 1) = 0;
              *((_DWORD *)v9 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v21 = (_QWORD *)*(v9 - 2);
          v22 = &v21[3 * *((unsigned int *)v9 - 2)];
          if ( v21 != v22 )
          {
            do
            {
              v23 = *(v22 - 1);
              v22 -= 3;
              if ( v23 != 0 && v23 != -8 && v23 != -16 )
                sub_1649B30(v22);
            }
            while ( v21 != v22 );
            v22 = (_QWORD *)*(v9 - 2);
          }
          if ( v9 != v22 )
            _libc_free((unsigned __int64)v22);
        }
        if ( v7 == v9 + 6 )
          break;
        v9 += 9;
      }
    }
    return (_QWORD *)j___libc_free_0(v37);
  }
  else
  {
    v32 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[9 * v32]; k != result; result += 9 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
