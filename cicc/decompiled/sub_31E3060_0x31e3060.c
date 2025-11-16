// Function: sub_31E3060
// Address: 0x31e3060
//
_QWORD *__fastcall sub_31E3060(__int64 a1, int a2)
{
  unsigned __int64 v2; // rcx
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned int v6; // eax
  _QWORD *result; // rax
  __int64 v8; // rcx
  _QWORD *v9; // r12
  _QWORD *i; // rcx
  _QWORD *v11; // rbx
  __int64 j; // rax
  __int64 v13; // r15
  int v14; // eax
  int v15; // esi
  __int64 v16; // rdi
  unsigned int v17; // ecx
  _QWORD *v18; // rax
  __int64 v19; // r8
  unsigned __int64 v20; // rdi
  __int64 v21; // rdx
  _QWORD *k; // rdx
  int v23; // r10d
  _QWORD *v24; // r9
  __int64 v25; // rcx
  _QWORD *v26; // [rsp+0h] [rbp-80h]
  _QWORD *v27; // [rsp+0h] [rbp-80h]
  __int64 v28; // [rsp+8h] [rbp-78h]
  _QWORD v29[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v30; // [rsp+20h] [rbp-60h]
  _QWORD v31[2]; // [rsp+30h] [rbp-50h] BYREF
  __int64 v32; // [rsp+40h] [rbp-40h]

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
    v28 = 48 * v4;
    v9 = (_QWORD *)(v5 + 48 * v4);
    for ( i = &result[6 * v8]; i != result; result += 6 )
    {
      if ( result )
      {
        *result = 0;
        result[1] = 0;
        result[2] = -4096;
      }
    }
    v29[0] = 0;
    v29[1] = 0;
    v30 = -4096;
    v31[0] = 0;
    v31[1] = 0;
    v32 = -8192;
    if ( v9 != (_QWORD *)v5 )
    {
      v11 = (_QWORD *)v5;
      for ( j = -4096; ; j = v30 )
      {
        v13 = v11[2];
        if ( v13 != j )
        {
          j = v32;
          if ( v13 != v32 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = (_QWORD *)(v16 + 48LL * v17);
            v19 = v18[2];
            if ( v19 != v13 )
            {
              v23 = 1;
              v24 = 0;
              while ( v19 != -4096 )
              {
                if ( !v24 && v19 == -8192 )
                  v24 = v18;
                v17 = v15 & (v23 + v17);
                v18 = (_QWORD *)(v16 + 48LL * v17);
                v19 = v18[2];
                if ( v13 == v19 )
                  goto LABEL_15;
                ++v23;
              }
              if ( v24 )
              {
                v25 = v24[2];
                v18 = v24;
              }
              else
              {
                v25 = v18[2];
              }
              if ( v13 != v25 )
              {
                if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
                {
                  v26 = v18;
                  sub_BD60C0(v18);
                  v18 = v26;
                }
                v18[2] = v13;
                if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
                {
                  v27 = v18;
                  sub_BD73F0((__int64)v18);
                  v18 = v27;
                }
              }
            }
LABEL_15:
            v18[3] = v11[3];
            v18[4] = v11[4];
            v18[5] = v11[5];
            v11[3] = 0;
            v11[5] = 0;
            v11[4] = 0;
            ++*(_DWORD *)(a1 + 16);
            v20 = v11[3];
            if ( v20 )
              j_j___libc_free_0(v20);
            j = v11[2];
          }
        }
        if ( j != -4096 && j != 0 && j != -8192 )
          sub_BD60C0(v11);
        v11 += 6;
        if ( v9 == v11 )
          break;
      }
      if ( v32 != 0 && v32 != -4096 && v32 != -8192 )
        sub_BD60C0(v31);
      if ( v30 != -8192 && v30 != -4096 && v30 != 0 )
        sub_BD60C0(v29);
    }
    return (_QWORD *)sub_C7D6A0(v5, v28, 8);
  }
  else
  {
    v21 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    for ( k = &result[6 * v21]; k != result; result += 6 )
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
