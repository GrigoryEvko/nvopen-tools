// Function: sub_1378D70
// Address: 0x1378d70
//
_QWORD *__fastcall sub_1378D70(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r12
  unsigned __int64 v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r15
  __int64 v8; // rdx
  _QWORD *i; // rcx
  char v10; // dl
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // r14
  __int64 v19; // rsi
  __int64 v20; // rdx
  _QWORD *j; // rcx
  char v22; // dl
  int v23; // r9d
  __int64 v24; // r8
  __int64 v25; // rax
  __int64 v26; // rdi
  _QWORD v27[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v28; // [rsp+28h] [rbp-78h]
  __int64 v29; // [rsp+30h] [rbp-70h]
  void *v30; // [rsp+40h] [rbp-60h]
  _QWORD v31[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v32; // [rsp+58h] [rbp-48h]
  __int64 v33; // [rsp+60h] [rbp-40h]

  v3 = *(unsigned int *)(a1 + 24);
  v4 = *(_QWORD **)(a1 + 8);
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
    v8 = *(unsigned int *)(a1 + 24);
    v31[0] = 2;
    v33 = 0;
    for ( i = &result[5 * v8]; i != result; result += 5 )
    {
      if ( result )
      {
        v10 = v31[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49E8A80;
        result[1] = v10 & 6;
        result[4] = v33;
      }
    }
    v27[1] = 0;
    v11 = v4;
    v30 = &unk_49E8A80;
    v12 = -8;
    v27[0] = 2;
    v28 = -8;
    v29 = 0;
    v31[0] = 2;
    v31[1] = 0;
    v32 = -16;
    v33 = 0;
    if ( v7 != v4 )
    {
      while ( 1 )
      {
        v13 = v11[3];
        if ( v13 != v12 )
        {
          v12 = v32;
          if ( v13 != v32 )
          {
            v14 = *(_DWORD *)(a1 + 24);
            if ( !v14 )
              BUG();
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 8);
            v17 = (v14 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
            v18 = v16 + 40LL * v17;
            v19 = *(_QWORD *)(v18 + 24);
            if ( v13 != v19 )
            {
              v23 = 1;
              v24 = 0;
              while ( v19 != -8 )
              {
                if ( v19 == -16 && !v24 )
                  v24 = v18;
                v17 = v15 & (v23 + v17);
                v18 = v16 + 40LL * v17;
                v19 = *(_QWORD *)(v18 + 24);
                if ( v13 == v19 )
                  goto LABEL_15;
                ++v23;
              }
              if ( v24 )
              {
                v25 = *(_QWORD *)(v24 + 24);
                v18 = v24;
              }
              else
              {
                v25 = *(_QWORD *)(v18 + 24);
              }
              v26 = v18 + 8;
              if ( v13 != v25 )
              {
                if ( v25 != 0 && v25 != -8 && v25 != -16 )
                {
                  sub_1649B30(v26);
                  v13 = v11[3];
                  v26 = v18 + 8;
                }
                *(_QWORD *)(v18 + 24) = v13;
                if ( v13 != 0 && v13 != -8 && v13 != -16 )
                  sub_1649AC0(v26, v11[1] & 0xFFFFFFFFFFFFFFF8LL);
              }
            }
LABEL_15:
            *(_QWORD *)(v18 + 32) = v11[4];
            ++*(_DWORD *)(a1 + 16);
            v12 = v11[3];
          }
        }
        *v11 = &unk_49EE2B0;
        if ( v12 != -8 && v12 != 0 && v12 != -16 )
          sub_1649B30(v11 + 1);
        v11 += 5;
        if ( v7 == v11 )
          break;
        v12 = v28;
      }
      v30 = &unk_49EE2B0;
      if ( v32 != -8 && v32 != 0 && v32 != -16 )
        sub_1649B30(v31);
    }
    if ( v28 != -8 && v28 != 0 && v28 != -16 )
      sub_1649B30(v27);
    return (_QWORD *)j___libc_free_0(v4);
  }
  else
  {
    v20 = *(unsigned int *)(a1 + 24);
    *(_QWORD *)(a1 + 16) = 0;
    v31[0] = 2;
    v33 = 0;
    for ( j = &result[5 * v20]; j != result; result += 5 )
    {
      if ( result )
      {
        v22 = v31[0];
        result[2] = 0;
        result[3] = -8;
        *result = &unk_49E8A80;
        result[1] = v22 & 6;
        result[4] = v33;
      }
    }
  }
  return result;
}
