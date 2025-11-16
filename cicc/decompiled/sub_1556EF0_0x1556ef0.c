// Function: sub_1556EF0
// Address: 0x1556ef0
//
_QWORD *__fastcall sub_1556EF0(__int64 a1, int a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  unsigned __int64 v4; // rax
  _QWORD *result; // rax
  __int64 v6; // rdx
  __int64 v7; // r12
  _QWORD *i; // rsi
  char v9; // dl
  __int64 v10; // rbx
  __int64 j; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rsi
  __int64 v16; // r8
  __int64 v17; // rdx
  _QWORD *k; // rcx
  char v19; // dl
  __int64 v20; // [rsp+8h] [rbp-B8h]
  __int64 v21; // [rsp+10h] [rbp-B0h]
  __int64 v22; // [rsp+10h] [rbp-B0h]
  __int64 v23; // [rsp+28h] [rbp-98h] BYREF
  void *v24; // [rsp+30h] [rbp-90h]
  _QWORD v25[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v26; // [rsp+48h] [rbp-78h]
  __int64 v27; // [rsp+50h] [rbp-70h]
  void *v28; // [rsp+60h] [rbp-60h]
  _QWORD v29[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v30; // [rsp+78h] [rbp-48h]
  __int64 v31; // [rsp+80h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  v4 = ((((((((((unsigned int)(a2 - 1) | ((unsigned __int64)(unsigned int)(a2 - 1) >> 1)) >> 2)
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
  if ( (unsigned int)v4 < 0x40 )
    LODWORD(v4) = 64;
  *(_DWORD *)(a1 + 24) = v4;
  result = (_QWORD *)sub_22077B0(48LL * (unsigned int)v4);
  *(_QWORD *)(a1 + 8) = result;
  if ( v3 )
  {
    v6 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29[0] = 2;
    v31 = 0;
    v7 = v3 + 48 * v2;
    for ( i = &result[6 * v6]; i != result; result += 6 )
    {
      if ( result )
      {
        v9 = v29[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49ECE00;
        result[1] = v9 & 6;
        result[4] = v31;
      }
    }
    v25[1] = 0;
    v25[0] = 2;
    v26 = -8;
    v24 = &unk_49ECE00;
    v27 = 0;
    v29[0] = 2;
    v29[1] = 0;
    v30 = -16;
    v28 = &unk_49ECE00;
    v31 = 0;
    if ( v7 != v3 )
    {
      v10 = v3;
      for ( j = -8; ; j = v26 )
      {
        v12 = *(_QWORD *)(v10 + 24);
        if ( v12 != j )
        {
          j = v30;
          if ( v12 != v30 )
          {
            sub_154CE90(a1, v10, &v23);
            v13 = v23;
            v14 = *(_QWORD *)(v10 + 24);
            v15 = *(_QWORD *)(v23 + 24);
            if ( v15 != v14 )
            {
              v16 = v23 + 8;
              if ( v15 != -8 && v15 != 0 && v15 != -16 )
              {
                v20 = v23;
                v21 = v23 + 8;
                sub_1649B30(v23 + 8);
                v14 = *(_QWORD *)(v10 + 24);
                v13 = v20;
                v16 = v21;
              }
              *(_QWORD *)(v13 + 24) = v14;
              if ( v14 != -8 && v14 != 0 && v14 != -16 )
              {
                v22 = v13;
                sub_1649AC0(v16, *(_QWORD *)(v10 + 8) & 0xFFFFFFFFFFFFFFF8LL);
                v13 = v22;
              }
            }
            *(_QWORD *)(v13 + 32) = *(_QWORD *)(v10 + 32);
            *(_DWORD *)(v13 + 40) = *(_DWORD *)(v10 + 40);
            ++*(_DWORD *)(a1 + 16);
            j = *(_QWORD *)(v10 + 24);
          }
        }
        *(_QWORD *)v10 = &unk_49EE2B0;
        if ( j != -8 && j != 0 && j != -16 )
          sub_1649B30(v10 + 8);
        v10 += 48;
        if ( v7 == v10 )
          break;
      }
      v28 = &unk_49EE2B0;
      if ( v30 != -8 && v30 != 0 && v30 != -16 )
        sub_1649B30(v29);
    }
    v24 = &unk_49EE2B0;
    if ( v26 != -8 && v26 != 0 && v26 != -16 )
      sub_1649B30(v25);
    return (_QWORD *)j___libc_free_0(v3);
  }
  else
  {
    v17 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v29[0] = 2;
    v31 = 0;
    for ( k = &result[6 * v17]; k != result; result += 6 )
    {
      if ( result )
      {
        v19 = v29[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49ECE00;
        result[1] = v19 & 6;
        result[4] = v31;
      }
    }
  }
  return result;
}
