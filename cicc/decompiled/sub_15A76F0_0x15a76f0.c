// Function: sub_15A76F0
// Address: 0x15a76f0
//
_QWORD *__fastcall sub_15A76F0(__int64 a1, int a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r13
  _QWORD *i; // rdx
  _QWORD *v10; // r15
  __int64 v11; // rax
  int v12; // edx
  int v13; // esi
  __int64 v14; // rdi
  int v15; // r10d
  __int64 *v16; // r9
  unsigned int v17; // edx
  __int64 *v18; // r12
  __int64 v19; // r8
  __int64 *v20; // rdx
  unsigned int v21; // r9d
  __int64 v22; // r14
  unsigned __int64 v23; // r12
  __int64 v24; // rsi
  _QWORD *v25; // r14
  __int64 *v26; // rdi
  __int64 v27; // rax
  __int64 *j; // r10
  __int64 v29; // rsi
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rsi
  __int64 v33; // rdx
  _QWORD *k; // rdx
  __int64 *v35; // [rsp+8h] [rbp-58h]
  unsigned int v36; // [rsp+14h] [rbp-4Ch]
  __int64 *v37; // [rsp+18h] [rbp-48h]
  unsigned int v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  _QWORD *v40; // [rsp+28h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
  v40 = v4;
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
  result = (_QWORD *)sub_22077B0(32LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    v7 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v8 = &v4[4 * v3];
    for ( i = &result[4 * v7]; i != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
    v10 = v4 + 3;
    if ( v8 != v4 )
    {
      while ( 1 )
      {
        v11 = *(v10 - 3);
        if ( v11 != -16 && v11 != -8 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(v10 - 3);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
          v18 = (__int64 *)(v14 + 32LL * v17);
          v19 = *v18;
          if ( v11 != *v18 )
          {
            while ( v19 != -8 )
            {
              if ( !v16 && v19 == -16 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 32LL * v17);
              v19 = *v18;
              if ( v11 == *v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          *v18 = v11;
          v20 = v18 + 3;
          v18[2] = 0x100000000LL;
          v18[1] = (__int64)(v18 + 3);
          v21 = *((_DWORD *)v10 - 2);
          if ( v18 + 1 != v10 - 2 && v21 )
          {
            v25 = (_QWORD *)*(v10 - 2);
            if ( v10 == v25 )
            {
              if ( v21 == 1 )
              {
                v26 = v10;
                v27 = 1;
              }
              else
              {
                v38 = *((_DWORD *)v10 - 2);
                sub_15A6A10((__int64)(v18 + 1), v21);
                v20 = (__int64 *)v18[1];
                v26 = (__int64 *)*(v10 - 2);
                v27 = *((unsigned int *)v10 - 2);
                v21 = v38;
              }
              for ( j = &v26[v27]; j != v26; ++v20 )
              {
                if ( v20 )
                {
                  v29 = *v26;
                  *v20 = *v26;
                  if ( v29 )
                  {
                    v35 = j;
                    v36 = v21;
                    v37 = v20;
                    sub_1623210(v26, v29, v20);
                    j = v35;
                    v21 = v36;
                    v20 = v37;
                    *v26 = 0;
                  }
                }
                ++v26;
              }
              *((_DWORD *)v18 + 4) = v21;
              v30 = *(v25 - 2);
              v31 = v30 + 8LL * *((unsigned int *)v25 - 2);
              while ( v30 != v31 )
              {
                v32 = *(_QWORD *)(v31 - 8);
                v31 -= 8;
                if ( v32 )
                {
                  v39 = v30;
                  sub_161E7C0(v31);
                  v30 = v39;
                }
              }
              *((_DWORD *)v25 - 2) = 0;
            }
            else
            {
              v18[1] = (__int64)v25;
              *((_DWORD *)v18 + 4) = *((_DWORD *)v10 - 2);
              *((_DWORD *)v18 + 5) = *((_DWORD *)v10 - 1);
              *(v10 - 2) = v10;
              *((_DWORD *)v10 - 1) = 0;
              *((_DWORD *)v10 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = *(v10 - 2);
          v23 = v22 + 8LL * *((unsigned int *)v10 - 2);
          if ( v22 != v23 )
          {
            do
            {
              v24 = *(_QWORD *)(v23 - 8);
              v23 -= 8LL;
              if ( v24 )
                sub_161E7C0(v23);
            }
            while ( v22 != v23 );
            v23 = *(v10 - 2);
          }
          if ( v10 != (_QWORD *)v23 )
            _libc_free(v23);
        }
        if ( v8 == v10 + 1 )
          break;
        v10 += 4;
      }
    }
    return (_QWORD *)j___libc_free_0(v40);
  }
  else
  {
    v33 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[4 * v33]; k != result; result += 4 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
