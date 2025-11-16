// Function: sub_31E01F0
// Address: 0x31e01f0
//
_QWORD *__fastcall sub_31E01F0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  __int64 v9; // r12
  _QWORD *i; // rcx
  __int64 v11; // rbx
  __int64 j; // rax
  __int64 v13; // r15
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // ecx
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 *v22; // rax
  unsigned __int64 v23; // r15
  __int64 v24; // rdx
  _QWORD *k; // rdx
  int v26; // r10d
  __int64 v27; // r9
  __int64 v28; // rcx
  __int64 v29; // [rsp+0h] [rbp-80h]
  __int64 v30; // [rsp+0h] [rbp-80h]
  __int64 v31; // [rsp+8h] [rbp-78h]
  _QWORD v32[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v33; // [rsp+20h] [rbp-60h]
  _QWORD v34[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v35; // [rsp+40h] [rbp-40h]

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
  result = (_QWORD *)sub_C7D670(48LL * v6, 8);
  *(_QWORD *)(a1 + 8) = result;
  if ( v5 )
  {
    v8 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v31 = 48 * v4;
    v9 = v5 + 48 * v4;
    for ( i = &result[6 * v8]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v32[0] = 0;
    v32[1] = 0;
    v33 = -4096;
    v34[0] = 0;
    v34[1] = 0;
    v35 = -8192;
    if ( v9 != v5 )
    {
      v11 = v5;
      for ( j = -4096; ; j = v33 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        if ( v13 != j )
        {
          j = v35;
          if ( v13 != v35 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = v16 + 48LL * v17;
            v19 = *(_QWORD *)(v18 + 16);
            if ( v19 != v13 )
            {
              v26 = 1;
              v27 = 0;
              while ( v19 != -4096 )
              {
                if ( !v27 && v19 == -8192 )
                  v27 = v18;
                v17 = v15 & (v26 + v17);
                v18 = v16 + 48LL * v17;
                v19 = *(_QWORD *)(v18 + 16);
                if ( v13 == v19 )
                  goto LABEL_14;
                ++v26;
              }
              if ( v27 )
              {
                v28 = *(_QWORD *)(v27 + 16);
                v18 = v27;
              }
              else
              {
                v28 = *(_QWORD *)(v18 + 16);
              }
              if ( v13 != v28 )
              {
                if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
                {
                  v29 = v18;
                  sub_BD60C0((_QWORD *)v18);
                  v18 = v29;
                }
                *(_QWORD *)(v18 + 16) = v13;
                if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
                {
                  v30 = v18;
                  sub_BD73F0(v18);
                  v18 = v30;
                }
              }
            }
LABEL_14:
            *(_QWORD *)(v18 + 24) = *(_QWORD *)(v11 + 24);
            v20 = *(_QWORD *)(v11 + 32);
            *(_QWORD *)(v11 + 24) = 0;
            *(_QWORD *)(v18 + 32) = v20;
            *(_DWORD *)(v18 + 40) = *(_DWORD *)(v11 + 40);
            ++*(_DWORD *)(a1 + 16);
            v21 = *(_QWORD *)(v11 + 24);
            if ( v21 )
            {
              if ( (v21 & 4) != 0 )
              {
                v22 = (unsigned __int64 *)(v21 & 0xFFFFFFFFFFFFFFF8LL);
                v23 = (unsigned __int64)v22;
                if ( v22 )
                {
                  if ( (unsigned __int64 *)*v22 != v22 + 2 )
                    _libc_free(*v22);
                  j_j___libc_free_0(v23);
                }
              }
            }
            j = *(_QWORD *)(v11 + 16);
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0((_QWORD *)v11);
        v11 += 48;
        if ( v9 == v11 )
          break;
      }
      if ( v35 != -4096 && v35 != 0 && v35 != -8192 )
        sub_BD60C0(v34);
      if ( v33 != -8192 && v33 != -4096 && v33 != 0 )
        sub_BD60C0(v32);
    }
    return (_QWORD *)sub_C7D6A0(v5, v31, 8);
  }
  else
  {
    v24 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v24]; k != result; result += 6 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
  }
  return result;
}
