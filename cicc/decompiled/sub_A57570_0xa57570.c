// Function: sub_A57570
// Address: 0xa57570
//
_QWORD *__fastcall sub_A57570(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  __int64 v9; // r14
  __int64 v10; // rcx
  __int64 v11; // r12
  int v12; // esi
  int v13; // esi
  __int64 v14; // r9
  int v15; // r11d
  __int64 *v16; // r10
  unsigned int v17; // edi
  __int64 *v18; // r15
  __int64 v19; // r8
  __int64 v20; // rsi
  __int64 v21; // rcx
  _QWORD *v22; // rsi
  unsigned int v23; // r13d
  __int64 v24; // r13
  __int64 v25; // r8
  __int64 v26; // r15
  __int64 v27; // rdi
  __int64 v28; // rcx
  _QWORD *v29; // rcx
  __int64 v30; // rdi
  _QWORD *v31; // r9
  __int64 v32; // r15
  __int64 v33; // rdx
  __int64 v34; // r13
  __int64 v35; // rdi
  __int64 v36; // rcx
  _QWORD *j; // rdx
  __int64 v38; // [rsp+8h] [rbp-48h]
  __int64 v39; // [rsp+10h] [rbp-40h]
  __int64 v40; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v39 = v5;
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v38 = 56 * v4;
    v40 = 56 * v4 + v5;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v9 = v5 + 56;
    if ( v40 != v39 )
    {
      do
      {
        v10 = *(_QWORD *)(v9 - 56);
        v11 = v9;
        if ( v10 != -8192 && v10 != -4096 )
        {
          v12 = *(_DWORD *)(a1 + 24);
          if ( !v12 )
          {
            MEMORY[0] = *(_QWORD *)(v9 - 56);
            BUG();
          }
          v13 = v12 - 1;
          v14 = *(_QWORD *)(a1 + 8);
          v15 = 1;
          v16 = 0;
          v17 = v13 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v18 = (__int64 *)(v14 + 56LL * v17);
          v19 = *v18;
          if ( v10 != *v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = (__int64 *)(v14 + 56LL * v17);
              v19 = *v18;
              if ( v10 == *v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          v18[3] = 0;
          v18[2] = 0;
          *((_DWORD *)v18 + 8) = 0;
          *v18 = v10;
          v18[1] = 1;
          v20 = *(_QWORD *)(v9 - 40);
          ++*(_QWORD *)(v9 - 48);
          v21 = v18[2];
          v18[2] = v20;
          LODWORD(v20) = *(_DWORD *)(v9 - 32);
          *(_QWORD *)(v9 - 40) = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 6);
          *((_DWORD *)v18 + 6) = v20;
          LODWORD(v20) = *(_DWORD *)(v9 - 28);
          *(_DWORD *)(v9 - 32) = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 7);
          *((_DWORD *)v18 + 7) = v20;
          v22 = (_QWORD *)*(unsigned int *)(v9 - 24);
          *(_DWORD *)(v9 - 28) = v21;
          LODWORD(v21) = *((_DWORD *)v18 + 8);
          *((_DWORD *)v18 + 8) = (_DWORD)v22;
          *(_DWORD *)(v9 - 24) = v21;
          v18[5] = (__int64)(v18 + 7);
          v18[6] = 0;
          v23 = *(_DWORD *)(v9 - 8);
          if ( v23 && v18 + 5 != (__int64 *)(v9 - 16) )
          {
            v28 = *(_QWORD *)(v9 - 16);
            if ( v28 == v9 )
            {
              sub_A57470((__int64)(v18 + 5), v23);
              v29 = (_QWORD *)v18[5];
              v22 = *(_QWORD **)(v9 - 16);
              v30 = 4LL * *(unsigned int *)(v9 - 8);
              v31 = &v29[v30];
              if ( v30 * 8 )
              {
                do
                {
                  if ( v29 )
                  {
                    *v29 = *v22;
                    v29[1] = v22[1];
                    v29[2] = v22[2];
                    v29[3] = v22[3];
                    v22[3] = 0;
                    v22[2] = 0;
                    v22[1] = 0;
                  }
                  v29 += 4;
                  v22 += 4;
                }
                while ( v29 != v31 );
              }
              *((_DWORD *)v18 + 12) = v23;
              v32 = *(_QWORD *)(v9 - 16);
              v33 = 32LL * *(unsigned int *)(v9 - 8);
              v34 = v32 + v33;
              while ( v32 != v34 )
              {
                v35 = *(_QWORD *)(v34 - 24);
                v34 -= 32;
                if ( v35 )
                {
                  v22 = (_QWORD *)(*(_QWORD *)(v34 + 24) - v35);
                  j_j___libc_free_0(v35, v22);
                }
              }
              *(_DWORD *)(v9 - 8) = 0;
            }
            else
            {
              v18[5] = v28;
              *((_DWORD *)v18 + 12) = *(_DWORD *)(v9 - 8);
              *((_DWORD *)v18 + 13) = *(_DWORD *)(v9 - 4);
              *(_QWORD *)(v9 - 16) = v9;
              *(_DWORD *)(v9 - 4) = 0;
              *(_DWORD *)(v9 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v24 = *(_QWORD *)(v9 - 16);
          v25 = 32LL * *(unsigned int *)(v9 - 8);
          v26 = v24 + v25;
          if ( v24 != v24 + v25 )
          {
            do
            {
              v27 = *(_QWORD *)(v26 - 24);
              v26 -= 32;
              if ( v27 )
              {
                v22 = (_QWORD *)(*(_QWORD *)(v26 + 24) - v27);
                j_j___libc_free_0(v27, v22);
              }
            }
            while ( v24 != v26 );
            v26 = *(_QWORD *)(v9 - 16);
          }
          if ( v26 != v9 )
            _libc_free(v26, v22);
          sub_C7D6A0(*(_QWORD *)(v9 - 40), 16LL * *(unsigned int *)(v9 - 24), 8);
        }
        v9 += 56;
      }
      while ( v40 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v39, v38, 8);
  }
  else
  {
    v36 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[7 * v36]; j != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
