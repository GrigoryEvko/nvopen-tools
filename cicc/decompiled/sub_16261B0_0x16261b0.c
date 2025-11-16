// Function: sub_16261B0
// Address: 0x16261b0
//
_QWORD *__fastcall sub_16261B0(__int64 a1, int a2)
{
  __int64 v3; // r12
  _QWORD *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r13
  _QWORD *i; // rdx
  _QWORD *v9; // r15
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // r8
  int v14; // r10d
  unsigned int *v15; // r9
  unsigned int v16; // edx
  unsigned int *v17; // r12
  __int64 v18; // rdi
  __int64 *v19; // r8
  unsigned int v20; // r10d
  __int64 v21; // r14
  unsigned __int64 v22; // r12
  __int64 v23; // rsi
  _QWORD *v24; // r14
  _QWORD *v25; // rax
  __int64 v26; // r11
  _QWORD *v27; // rdi
  _QWORD *v28; // r11
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rsi
  unsigned __int8 *v32; // rsi
  __int64 v33; // rdx
  _QWORD *j; // rdx
  _QWORD *v35; // [rsp+8h] [rbp-58h]
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
  result = (_QWORD *)sub_22077B0(40LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = result;
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v7 = &v4[5 * v3];
    for ( i = &result[5 * *(unsigned int *)(a1 + 24)]; i != result; result += 5 )
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
          v17 = (unsigned int *)(v13 + 40LL * v16);
          v18 = *(_QWORD *)v17;
          if ( v10 != *(_QWORD *)v17 )
          {
            while ( v18 != -8 )
            {
              if ( !v15 && v18 == -16 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = (unsigned int *)(v13 + 40LL * v16);
              v18 = *(_QWORD *)v17;
              if ( v10 == *(_QWORD *)v17 )
                goto LABEL_13;
              ++v14;
            }
            if ( v15 )
              v17 = v15;
          }
LABEL_13:
          *(_QWORD *)v17 = v10;
          v19 = (__int64 *)(v17 + 6);
          *((_QWORD *)v17 + 1) = v17 + 6;
          *((_QWORD *)v17 + 2) = 0x100000000LL;
          v20 = *((_DWORD *)v9 - 2);
          if ( v20 && v17 + 2 != (unsigned int *)(v9 - 2) )
          {
            v24 = (_QWORD *)*(v9 - 2);
            if ( v9 == v24 )
            {
              if ( v20 == 1 )
              {
                v25 = v9;
                v26 = 1;
              }
              else
              {
                v38 = *((_DWORD *)v9 - 2);
                sub_1623260(v17 + 2, v20);
                v19 = (__int64 *)*((_QWORD *)v17 + 1);
                v25 = (_QWORD *)*(v9 - 2);
                v26 = *((unsigned int *)v9 - 2);
                v20 = v38;
              }
              v27 = v25 + 1;
              v28 = &v25[2 * v26];
              if ( v28 != v25 )
              {
                while ( 1 )
                {
                  if ( v19 )
                  {
                    *(_DWORD *)v19 = *((_DWORD *)v27 - 2);
                    v32 = (unsigned __int8 *)*v27;
                    v19[1] = *v27;
                    if ( v32 )
                    {
                      v35 = v28;
                      v36 = v20;
                      v37 = v19;
                      sub_1623210((__int64)v27, v32, (__int64)(v19 + 1));
                      v28 = v35;
                      v20 = v36;
                      v19 = v37;
                      *v27 = 0;
                    }
                  }
                  v19 += 2;
                  if ( v28 == v27 + 1 )
                    break;
                  v27 += 2;
                }
              }
              v17[4] = v20;
              v29 = *(v24 - 2);
              v30 = v29 + 16LL * *((unsigned int *)v24 - 2);
              while ( v29 != v30 )
              {
                v31 = *(_QWORD *)(v30 - 8);
                v30 -= 16;
                if ( v31 )
                {
                  v39 = v29;
                  sub_161E7C0(v30 + 8, v31);
                  v29 = v39;
                }
              }
              *((_DWORD *)v24 - 2) = 0;
            }
            else
            {
              *((_QWORD *)v17 + 1) = v24;
              v17[4] = *((_DWORD *)v9 - 2);
              v17[5] = *((_DWORD *)v9 - 1);
              *(v9 - 2) = v9;
              *((_DWORD *)v9 - 1) = 0;
              *((_DWORD *)v9 - 2) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v21 = *(v9 - 2);
          v22 = v21 + 16LL * *((unsigned int *)v9 - 2);
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
            v22 = *(v9 - 2);
          }
          if ( (_QWORD *)v22 != v9 )
            _libc_free(v22);
        }
        if ( v7 == v9 + 2 )
          break;
        v9 += 5;
      }
    }
    return (_QWORD *)j___libc_free_0(v40);
  }
  else
  {
    v33 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v33]; j != result; result += 5 )
    {
      if ( result )
        *result = -8;
    }
  }
  return result;
}
