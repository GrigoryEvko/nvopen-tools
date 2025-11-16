// Function: sub_1624590
// Address: 0x1624590
//
_QWORD *__fastcall sub_1624590(__int64 a1, int a2)
{
  __int64 v3; // r12
  __int64 v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  __int64 v7; // r13
  _QWORD *i; // rdx
  __int64 v9; // r15
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  unsigned int v15; // edx
  unsigned int *v16; // r9
  unsigned int *v17; // r12
  __int64 v18; // rdi
  __int64 *v19; // r8
  unsigned int v20; // r11d
  __int64 v21; // r14
  unsigned __int64 v22; // r12
  __int64 v23; // rsi
  __int64 v24; // r14
  __int64 v25; // rsi
  __int64 v26; // rax
  __int64 v27; // rsi
  _QWORD *v28; // rdi
  _QWORD *v29; // r10
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rsi
  unsigned __int8 *v33; // rsi
  __int64 v34; // rcx
  _QWORD *j; // rdx
  _QWORD *v36; // [rsp+8h] [rbp-58h]
  unsigned int v37; // [rsp+14h] [rbp-4Ch]
  __int64 *v38; // [rsp+18h] [rbp-48h]
  unsigned int v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v41 = v4;
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
  result = (_QWORD *)sub_22077B0(56LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = v4 + 56 * v3;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
    v9 = v4 + 24;
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v9 - 24);
        if ( v10 != -16 && v10 != -8 )
        {
          v11 = *(_DWORD *)(a1 + 24);
          if ( !v11 )
          {
            MEMORY[0] = *(_QWORD *)(v9 - 24);
            BUG();
          }
          v12 = v11 - 1;
          v13 = *(_QWORD *)(a1 + 8);
          v14 = 1;
          v15 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v16 = 0;
          v17 = (unsigned int *)(v13 + 56LL * v15);
          v18 = *(_QWORD *)v17;
          if ( v10 != *(_QWORD *)v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v16 && v18 == -16 )
                v16 = v17;
              v15 = v12 & (v14 + v15);
              v17 = (unsigned int *)(v13 + 56LL * v15);
              v18 = *(_QWORD *)v17;
              if ( v10 == *(_QWORD *)v17 )
                goto LABEL_13;
              ++v14;
            }
            if ( v16 )
              v17 = v16;
          }
LABEL_13:
          *(_QWORD *)v17 = v10;
          v19 = (__int64 *)(v17 + 6);
          *((_QWORD *)v17 + 1) = v17 + 6;
          *((_QWORD *)v17 + 2) = 0x200000000LL;
          v20 = *(_DWORD *)(v9 - 8);
          if ( v20 && v17 + 2 != (unsigned int *)(v9 - 16) )
          {
            v24 = *(_QWORD *)(v9 - 16);
            if ( v9 == v24 )
            {
              v25 = v20;
              v26 = v9;
              if ( v20 > 2 )
              {
                v39 = *(_DWORD *)(v9 - 8);
                sub_1623400(v17 + 2, v20);
                v19 = (__int64 *)*((_QWORD *)v17 + 1);
                v26 = *(_QWORD *)(v9 - 16);
                v25 = *(unsigned int *)(v9 - 8);
                v20 = v39;
              }
              v27 = 16 * v25;
              v28 = (_QWORD *)(v26 + 8);
              v29 = (_QWORD *)(v26 + v27);
              if ( v26 + v27 != v26 )
              {
                while ( 1 )
                {
                  if ( v19 )
                  {
                    *(_DWORD *)v19 = *((_DWORD *)v28 - 2);
                    v33 = (unsigned __int8 *)*v28;
                    v19[1] = *v28;
                    if ( v33 )
                    {
                      v36 = v29;
                      v37 = v20;
                      v38 = v19;
                      sub_1623210((__int64)v28, v33, (__int64)(v19 + 1));
                      v29 = v36;
                      v20 = v37;
                      v19 = v38;
                      *v28 = 0;
                    }
                  }
                  v19 += 2;
                  if ( v29 == v28 + 1 )
                    break;
                  v28 += 2;
                }
              }
              v17[4] = v20;
              v30 = *(_QWORD *)(v24 - 16);
              v31 = v30 + 16LL * *(unsigned int *)(v24 - 8);
              while ( v30 != v31 )
              {
                v32 = *(_QWORD *)(v31 - 8);
                v31 -= 16;
                if ( v32 )
                {
                  v40 = v30;
                  sub_161E7C0(v31 + 8, v32);
                  v30 = v40;
                }
              }
              *(_DWORD *)(v24 - 8) = 0;
            }
            else
            {
              *((_QWORD *)v17 + 1) = v24;
              v17[4] = *(_DWORD *)(v9 - 8);
              v17[5] = *(_DWORD *)(v9 - 4);
              *(_QWORD *)(v9 - 16) = v9;
              *(_DWORD *)(v9 - 4) = 0;
              *(_DWORD *)(v9 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v21 = *(_QWORD *)(v9 - 16);
          v22 = v21 + 16LL * *(unsigned int *)(v9 - 8);
          if ( v21 != v22 )
          {
            do
            {
              v23 = *(_QWORD *)(v22 - 8);
              v22 -= 16LL;
              if ( v23 )
                sub_161E7C0(v22 + 8, v23);
            }
            while ( v21 != v22 );
            v22 = *(_QWORD *)(v9 - 16);
          }
          if ( v22 != v9 )
            _libc_free(v22);
        }
        if ( v7 == v9 + 32 )
          break;
        v9 += 56;
      }
    }
    return (_QWORD *)j___libc_free_0(v41);
  }
  else
  {
    v34 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v34]; j != result; result += 7 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
