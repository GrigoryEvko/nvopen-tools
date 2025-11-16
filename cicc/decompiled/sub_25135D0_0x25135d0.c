// Function: sub_25135D0
// Address: 0x25135d0
//
_QWORD *__fastcall sub_25135D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // rbx
  __int64 v5; // r15
  unsigned int v6; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  __int64 v9; // rbx
  __int64 v10; // rax
  int v11; // edx
  int v12; // esi
  __int64 v13; // rdi
  int v14; // r10d
  __int64 v15; // r9
  unsigned int v16; // ecx
  __int64 v17; // rdx
  __int64 v18; // r8
  _QWORD *v19; // rax
  __int64 v20; // rcx
  unsigned int v21; // r15d
  __int64 v22; // r12
  unsigned __int64 v23; // r15
  _QWORD *v24; // r14
  void (__fastcall *v25)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // rax
  void (__fastcall *v26)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // rax
  unsigned __int64 v27; // rdi
  __int64 v28; // r12
  __int64 v29; // rsi
  _QWORD *v30; // rcx
  __int64 v31; // rsi
  _QWORD *v32; // rdi
  __int64 v33; // r15
  _QWORD *v34; // r14
  void (__fastcall *v35)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64, __int64); // rax
  void (__fastcall *v36)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64); // rax
  unsigned __int64 v37; // rdi
  __int64 v38; // rdx
  _QWORD *j; // rdx
  __int64 v40; // [rsp+0h] [rbp-50h]
  __int64 v41; // [rsp+8h] [rbp-48h]
  __int64 v42; // [rsp+8h] [rbp-48h]
  __int64 v43; // [rsp+10h] [rbp-40h]
  __int64 v44; // [rsp+18h] [rbp-38h]

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v43 = v5;
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
  result = (_QWORD *)sub_C7D670(88LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v40 = 88 * v4;
    v44 = 88 * v4 + v5;
    for ( i = &result[11 * *(unsigned int *)(a1 + 24)]; i != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
    v9 = v5 + 24;
    if ( v44 != v5 )
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(v9 - 24);
        if ( v10 != -8192 && v10 != -4096 )
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
          v15 = 0;
          v16 = (v11 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
          v17 = v13 + 88LL * v16;
          v18 = *(_QWORD *)v17;
          if ( v10 != *(_QWORD *)v17 )
          {
            while ( v18 != -4096 )
            {
              if ( !v15 && v18 == -8192 )
                v15 = v17;
              v16 = v12 & (v14 + v16);
              v17 = v13 + 88LL * v16;
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
          v19 = (_QWORD *)(v17 + 24);
          *(_QWORD *)(v17 + 16) = 0x800000000LL;
          v20 = v9 - 16;
          *(_QWORD *)(v17 + 8) = v17 + 24;
          v21 = *(_DWORD *)(v9 - 8);
          if ( v17 + 8 != v9 - 16 && v21 )
          {
            v28 = *(_QWORD *)(v9 - 16);
            if ( v9 == v28 )
            {
              v29 = v21;
              v30 = (_QWORD *)v9;
              if ( v21 > 8 )
              {
                v41 = v17;
                sub_25126F0(v17 + 8, v21, v17, v9, v18, v15);
                v17 = v41;
                v30 = *(_QWORD **)(v9 - 16);
                v29 = *(unsigned int *)(v9 - 8);
                v19 = *(_QWORD **)(v41 + 8);
              }
              v31 = v29;
              v32 = &v19[v31];
              if ( v31 * 8 )
              {
                do
                {
                  if ( v19 )
                  {
                    *v19 = *v30;
                    *v30 = 0;
                  }
                  ++v19;
                  ++v30;
                }
                while ( v19 != v32 );
              }
              *(_DWORD *)(v17 + 16) = v21;
              v20 = *(_QWORD *)(v28 - 16);
              v42 = v20;
              v33 = v20 + 8LL * *(unsigned int *)(v28 - 8);
              if ( v20 != v33 )
              {
                do
                {
                  v34 = *(_QWORD **)(v33 - 8);
                  v33 -= 8;
                  if ( v34 )
                  {
                    v35 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64, __int64))v34[19];
                    if ( v35 )
                      v35(v34 + 17, v34 + 17, 3, v20, v18, v15, v40);
                    v36 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64))v34[15];
                    if ( v36 )
                      v36(v34 + 13, v34 + 13, 3, v20, v18, v15);
                    v37 = v34[3];
                    if ( (_QWORD *)v37 != v34 + 5 )
                      _libc_free(v37);
                    j_j___libc_free_0((unsigned __int64)v34);
                  }
                }
                while ( v42 != v33 );
              }
              *(_DWORD *)(v28 - 8) = 0;
            }
            else
            {
              *(_QWORD *)(v17 + 8) = v28;
              *(_DWORD *)(v17 + 16) = *(_DWORD *)(v9 - 8);
              *(_DWORD *)(v17 + 20) = *(_DWORD *)(v9 - 4);
              *(_QWORD *)(v9 - 16) = v9;
              *(_DWORD *)(v9 - 4) = 0;
              *(_DWORD *)(v9 - 8) = 0;
            }
          }
          ++*(_DWORD *)(a1 + 16);
          v22 = *(_QWORD *)(v9 - 16);
          v23 = v22 + 8LL * *(unsigned int *)(v9 - 8);
          if ( v22 != v23 )
          {
            do
            {
              v24 = *(_QWORD **)(v23 - 8);
              v23 -= 8LL;
              if ( v24 )
              {
                v25 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64))v24[19];
                if ( v25 )
                  v25(v24 + 17, v24 + 17, 3, v20, v18, v15);
                v26 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64, __int64, __int64, __int64))v24[15];
                if ( v26 )
                  v26(v24 + 13, v24 + 13, 3, v20, v18, v15);
                v27 = v24[3];
                if ( (_QWORD *)v27 != v24 + 5 )
                  _libc_free(v27);
                j_j___libc_free_0((unsigned __int64)v24);
              }
            }
            while ( v22 != v23 );
            v23 = *(_QWORD *)(v9 - 16);
          }
          if ( v9 != v23 )
            _libc_free(v23);
        }
        if ( v44 == v9 + 64 )
          break;
        v9 += 88;
      }
    }
    return (_QWORD *)sub_C7D6A0(v43, v40, 8);
  }
  else
  {
    v38 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( j = &result[11 * v38]; j != result; result += 11 )
    {
      if ( result )
        *result = -4096;
    }
  }
  return result;
}
