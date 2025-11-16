// Function: sub_298C3B0
// Address: 0x298c3b0
//
_QWORD *__fastcall sub_298C3B0(__int64 a1, int a2)
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
  __int64 v16; // r10
  unsigned int v17; // edi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rsi
  __int64 v21; // rcx
  unsigned int v22; // r13d
  __int64 v23; // r13
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  __int64 v26; // r15
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r10
  __int64 v32; // rsi
  __int64 v33; // rsi
  __int64 v34; // r13
  unsigned __int64 v35; // rdi
  __int64 v36; // rcx
  _QWORD *j; // rdx
  __int64 v38; // [rsp+8h] [rbp-68h]
  __int64 v39; // [rsp+10h] [rbp-60h]
  __int64 v40; // [rsp+18h] [rbp-58h]
  __int64 v41; // [rsp+20h] [rbp-50h]
  __int64 v42; // [rsp+28h] [rbp-48h]
  __int64 v43; // [rsp+28h] [rbp-48h]
  __int64 v44; // [rsp+30h] [rbp-40h]
  __int64 v45; // [rsp+38h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v41 = 56 * v4;
    v45 = 56 * v4 + v5;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
        *result = -4096;
    }
    v9 = v5 + 56;
    if ( v45 != v44 )
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
          v18 = v14 + 56LL * v17;
          v19 = *(_QWORD *)v18;
          if ( v10 != *(_QWORD *)v18 )
          {
            while ( v19 != -4096 )
            {
              if ( !v16 && v19 == -8192 )
                v16 = v18;
              v17 = v13 & (v15 + v17);
              v18 = v14 + 56LL * v17;
              v19 = *(_QWORD *)v18;
              if ( v10 == *(_QWORD *)v18 )
                goto LABEL_13;
              ++v15;
            }
            if ( v16 )
              v18 = v16;
          }
LABEL_13:
          *(_QWORD *)(v18 + 24) = 0;
          *(_QWORD *)(v18 + 16) = 0;
          *(_DWORD *)(v18 + 32) = 0;
          *(_QWORD *)v18 = v10;
          *(_QWORD *)(v18 + 8) = 1;
          v20 = *(_QWORD *)(v9 - 40);
          ++*(_QWORD *)(v9 - 48);
          v21 = *(_QWORD *)(v18 + 16);
          *(_QWORD *)(v18 + 16) = v20;
          LODWORD(v20) = *(_DWORD *)(v9 - 32);
          *(_QWORD *)(v9 - 40) = v21;
          LODWORD(v21) = *(_DWORD *)(v18 + 24);
          *(_DWORD *)(v18 + 24) = v20;
          LODWORD(v20) = *(_DWORD *)(v9 - 28);
          *(_DWORD *)(v9 - 32) = v21;
          LODWORD(v21) = *(_DWORD *)(v18 + 28);
          *(_DWORD *)(v18 + 28) = v20;
          LODWORD(v20) = *(_DWORD *)(v9 - 24);
          *(_DWORD *)(v9 - 28) = v21;
          LODWORD(v21) = *(_DWORD *)(v18 + 32);
          *(_DWORD *)(v18 + 32) = v20;
          *(_DWORD *)(v9 - 24) = v21;
          *(_QWORD *)(v18 + 40) = v18 + 56;
          *(_QWORD *)(v18 + 48) = 0;
          v22 = *(_DWORD *)(v9 - 8);
          if ( v22 && v18 + 40 != v9 - 16 )
          {
            v26 = *(_QWORD *)(v9 - 16);
            if ( v9 == v26 )
            {
              v42 = v18;
              sub_298C290(v18 + 40, v22, v18, v9 - 16, v19, v14);
              v28 = *(_QWORD *)(v9 - 16);
              v29 = v42;
              v30 = *(_QWORD *)(v42 + 40);
              v31 = v28 + 56LL * *(unsigned int *)(v9 - 8);
              if ( v28 != v31 )
              {
                do
                {
                  while ( 1 )
                  {
                    if ( v30 )
                    {
                      v32 = *(_QWORD *)v28;
                      *(_DWORD *)(v30 + 16) = 0;
                      *(_DWORD *)(v30 + 20) = 2;
                      *(_QWORD *)v30 = v32;
                      *(_QWORD *)(v30 + 8) = v30 + 24;
                      if ( *(_DWORD *)(v28 + 16) )
                        break;
                    }
                    v28 += 56;
                    v30 += 56;
                    if ( v31 == v28 )
                      goto LABEL_32;
                  }
                  v38 = v29;
                  v39 = v31;
                  v40 = v28;
                  v43 = v30;
                  sub_2988720(v30 + 8, v28 + 8, v29, v30, v28, v27);
                  v31 = v39;
                  v29 = v38;
                  v28 = v40 + 56;
                  v30 = v43 + 56;
                }
                while ( v39 != v40 + 56 );
              }
LABEL_32:
              *(_DWORD *)(v29 + 48) = v22;
              v33 = *(_QWORD *)(v26 - 16);
              v34 = v33 + 56LL * *(unsigned int *)(v26 - 8);
              while ( v33 != v34 )
              {
                v34 -= 56;
                v35 = *(_QWORD *)(v34 + 8);
                if ( v35 != v34 + 24 )
                  _libc_free(v35);
              }
              *(_DWORD *)(v26 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v18 + 40) = v26;
              *(_DWORD *)(v18 + 48) = *(_DWORD *)(v9 - 8);
              *(_DWORD *)(v18 + 52) = *(_DWORD *)(v9 - 4);
              *(_QWORD *)(v9 - 16) = v9;
              *(_DWORD *)(v9 - 4) = 0;
              *(_DWORD *)(v9 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v23 = *(_QWORD *)(v9 - 16);
          v24 = v23 + 56LL * *(unsigned int *)(v9 - 8);
          if ( v23 != v24 )
          {
            do
            {
              v24 -= 56LL;
              v25 = *(_QWORD *)(v24 + 8);
              if ( v25 != v24 + 24 )
                _libc_free(v25);
            }
            while ( v23 != v24 );
            v24 = *(_QWORD *)(v9 - 16);
          }
          if ( v9 != v24 )
            _libc_free(v24);
          sub_C7D6A0(*(_QWORD *)(v9 - 40), 16LL * *(unsigned int *)(v9 - 24), 8);
        }
        v9 += 56;
      }
      while ( v45 != v11 );
    }
    return (_QWORD *)sub_C7D6A0(v44, v41, 8);
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
