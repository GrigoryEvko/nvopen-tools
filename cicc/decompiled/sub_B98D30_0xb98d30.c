// Function: sub_B98D30
// Address: 0xb98d30
//
_QWORD *__fastcall sub_B98D30(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v3; // rbx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rdx
  _QWORD *v9; // r12
  _QWORD *i; // rdx
  _DWORD *v11; // r15
  __int64 v12; // rax
  int v13; // edx
  int v14; // ecx
  __int64 v15; // rdi
  int v16; // r9d
  __int64 v17; // r8
  unsigned int v18; // edx
  __int64 v19; // r13
  __int64 v20; // rsi
  __int64 v21; // r8
  unsigned int v22; // r10d
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  _QWORD *v25; // r14
  _QWORD *v26; // rax
  __int64 v27; // r9
  _QWORD *v28; // r9
  _QWORD *v29; // r12
  __int64 v30; // r13
  _QWORD *v31; // rbx
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r13
  __int64 v35; // rdx
  _QWORD *j; // rdx
  __int64 v37; // [rsp+0h] [rbp-60h]
  __int64 v38; // [rsp+8h] [rbp-58h]
  unsigned int v39; // [rsp+14h] [rbp-4Ch]
  __int64 v40; // [rsp+18h] [rbp-48h]
  unsigned int v41; // [rsp+20h] [rbp-40h]
  _QWORD *v42; // [rsp+20h] [rbp-40h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v3 = a1;
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v44 = v5;
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
  result = (_QWORD *)sub_C7D670(40LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v40 = 40 * v4;
    v9 = (_QWORD *)(40 * v4 + v5);
    for ( i = &result[5 * v8]; i != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
    v11 = (_DWORD *)(v5 + 24);
    if ( v9 != (_QWORD *)v5 )
    {
      while ( 1 )
      {
        v12 = *((_QWORD *)v11 - 3);
        if ( v12 != -8192 && v12 != -4096 )
        {
          v13 = *(_DWORD *)(v3 + 24);
          if ( !v13 )
          {
            MEMORY[0] = *((_QWORD *)v11 - 3);
            BUG();
          }
          v14 = v13 - 1;
          v15 = *(_QWORD *)(v3 + 8);
          v16 = 1;
          v17 = 0;
          v18 = (v13 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v19 = v15 + 40LL * v18;
          v20 = *(_QWORD *)v19;
          if ( v12 != *(_QWORD *)v19 )
          {
            while ( v20 != -4096 )
            {
              if ( !v17 && v20 == -8192 )
                v17 = v19;
              v18 = v14 & (v16 + v18);
              v19 = v15 + 40LL * v18;
              v20 = *(_QWORD *)v19;
              if ( v12 == *(_QWORD *)v19 )
                goto LABEL_13;
              ++v16;
            }
            if ( v17 )
              v19 = v17;
          }
LABEL_13:
          *(_QWORD *)v19 = v12;
          v21 = v19 + 24;
          *(_QWORD *)(v19 + 8) = v19 + 24;
          *(_QWORD *)(v19 + 16) = 0x100000000LL;
          v22 = *(v11 - 2);
          if ( v22 && (_DWORD *)(v19 + 8) != v11 - 4 )
          {
            v25 = (_QWORD *)*((_QWORD *)v11 - 2);
            if ( v11 == (_DWORD *)v25 )
            {
              v20 = v22;
              v26 = v11;
              v27 = 1;
              if ( v22 != 1 )
              {
                v41 = *(v11 - 2);
                sub_B97B00((__int64 *)(v19 + 8), v22);
                v21 = *(_QWORD *)(v19 + 8);
                v26 = (_QWORD *)*((_QWORD *)v11 - 2);
                v27 = (unsigned int)*(v11 - 2);
                v22 = v41;
              }
              v28 = &v26[2 * v27];
              if ( v28 != v26 )
              {
                v42 = v9;
                v29 = v28;
                v38 = v19;
                v30 = v21;
                v37 = v3;
                v31 = v26 + 1;
                v39 = v22;
                while ( 1 )
                {
                  if ( v30 )
                  {
                    *(_DWORD *)v30 = *((_DWORD *)v31 - 2);
                    v20 = *v31;
                    *(_QWORD *)(v30 + 8) = *v31;
                    if ( v20 )
                    {
                      sub_B976B0((__int64)v31, (unsigned __int8 *)v20, v30 + 8);
                      *v31 = 0;
                    }
                  }
                  v30 += 16;
                  if ( v29 == v31 + 1 )
                    break;
                  v31 += 2;
                }
                v9 = v42;
                v22 = v39;
                v19 = v38;
                v3 = v37;
              }
              *(_DWORD *)(v19 + 16) = v22;
              v32 = *(v25 - 2);
              if ( v32 != v32 + 16LL * *((unsigned int *)v25 - 2) )
              {
                v43 = v3;
                v33 = v32 + 16LL * *((unsigned int *)v25 - 2);
                v34 = *(v25 - 2);
                do
                {
                  v20 = *(_QWORD *)(v33 - 8);
                  v33 -= 16;
                  if ( v20 )
                    sub_B91220(v33 + 8, v20);
                }
                while ( v34 != v33 );
                v3 = v43;
              }
              *((_DWORD *)v25 - 2) = 0;
            }
            else
            {
              *(_QWORD *)(v19 + 8) = v25;
              *(_DWORD *)(v19 + 16) = *(v11 - 2);
              *(_DWORD *)(v19 + 20) = *(v11 - 1);
              *((_QWORD *)v11 - 2) = v11;
              *(v11 - 1) = 0;
              *(v11 - 2) = 0;
            }
          }
          ++*(_DWORD *)(v3 + 16);
          v23 = (_QWORD *)*((_QWORD *)v11 - 2);
          v24 = &v23[2 * (unsigned int)*(v11 - 2)];
          if ( v23 != v24 )
          {
            do
            {
              v20 = *(v24 - 1);
              v24 -= 2;
              if ( v20 )
                sub_B91220((__int64)(v24 + 1), v20);
            }
            while ( v23 != v24 );
            v24 = (_QWORD *)*((_QWORD *)v11 - 2);
          }
          if ( v24 != (_QWORD *)v11 )
            _libc_free(v24, v20);
        }
        if ( v9 == (_QWORD *)(v11 + 4) )
          break;
        v11 += 10;
      }
    }
    return (_QWORD *)sub_C7D6A0(v44, v40, 8);
  }
  else
  {
    v35 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[5 * v35]; j != result; result += 5 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
