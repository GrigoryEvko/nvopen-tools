// Function: sub_297DFD0
// Address: 0x297dfd0
//
_QWORD *__fastcall sub_297DFD0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rdx
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned int v6; // eax
  _QWORD *result; // rax
  _QWORD *i; // rdx
  __int64 j; // r13
  __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // r12
  _QWORD *v17; // r15
  unsigned __int64 v18; // rdi
  __int64 v19; // rcx
  _QWORD *k; // rdx
  __int64 v21; // [rsp+8h] [rbp-58h]
  __int64 v22; // [rsp+18h] [rbp-48h]
  _QWORD *v23; // [rsp+28h] [rbp-38h] BYREF

  v2 = (unsigned int)(a2 - 1);
  v4 = *(unsigned int *)(a1 + 24);
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
  result = (_QWORD *)sub_C7D670(56LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v21 = 56 * v4;
    v22 = 56 * v4 + v5;
    for ( i = &result[7 * *(unsigned int *)(a1 + 24)]; i != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
        result[2] = -4096;
      }
    }
    for ( j = v5; v22 != j; j += 56 )
    {
      v10 = *(_QWORD *)(j + 16);
      if ( v10 == -4096 )
      {
        if ( *(_QWORD *)(j + 8) != -4096 || *(_QWORD *)j != -4096 )
        {
LABEL_11:
          sub_297D7C0(a1, (__int64 *)j, &v23);
          v11 = v23;
          v23[2] = *(_QWORD *)(j + 16);
          v11[1] = *(_QWORD *)(j + 8);
          *v11 = *(_QWORD *)j;
          v12 = v23;
          v23[5] = 0;
          v12[4] = 0;
          *((_DWORD *)v12 + 12) = 0;
          v12[3] = 1;
          v13 = *(_QWORD *)(j + 32);
          ++*(_QWORD *)(j + 24);
          v14 = v12[4];
          v12[4] = v13;
          LODWORD(v13) = *(_DWORD *)(j + 40);
          *(_QWORD *)(j + 32) = v14;
          LODWORD(v14) = *((_DWORD *)v12 + 10);
          *((_DWORD *)v12 + 10) = v13;
          LODWORD(v13) = *(_DWORD *)(j + 44);
          *(_DWORD *)(j + 40) = v14;
          LODWORD(v14) = *((_DWORD *)v12 + 11);
          *((_DWORD *)v12 + 11) = v13;
          LODWORD(v13) = *(_DWORD *)(j + 48);
          *(_DWORD *)(j + 44) = v14;
          LODWORD(v14) = *((_DWORD *)v12 + 12);
          *((_DWORD *)v12 + 12) = v13;
          *(_DWORD *)(j + 48) = v14;
          ++*(_DWORD *)(a1 + 16);
          v15 = *(unsigned int *)(j + 48);
          if ( (_DWORD)v15 )
          {
            v16 = *(_QWORD **)(j + 32);
            v17 = &v16[11 * v15];
            do
            {
              if ( *v16 != -8192 && *v16 != -4096 )
              {
                v18 = v16[1];
                if ( (_QWORD *)v18 != v16 + 3 )
                  _libc_free(v18);
              }
              v16 += 11;
            }
            while ( v17 != v16 );
            v15 = *(unsigned int *)(j + 48);
          }
          sub_C7D6A0(*(_QWORD *)(j + 32), 88 * v15, 8);
        }
      }
      else if ( v10 != -8192 || *(_QWORD *)(j + 8) != -8192 || *(_QWORD *)j != -8192 )
      {
        goto LABEL_11;
      }
    }
    return (_QWORD *)sub_C7D6A0(v5, v21, 8);
  }
  else
  {
    v19 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[7 * v19]; k != result; result += 7 )
    {
      if ( result )
      {
        *result = -4096;
        result[1] = -4096;
        result[2] = -4096;
      }
    }
  }
  return result;
}
