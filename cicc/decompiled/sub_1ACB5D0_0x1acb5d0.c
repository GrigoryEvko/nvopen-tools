// Function: sub_1ACB5D0
// Address: 0x1acb5d0
//
unsigned __int64 __fastcall sub_1ACB5D0(_QWORD *a1)
{
  unsigned int v1; // eax
  _BOOL8 v2; // rsi
  unsigned __int64 v3; // rdx
  unsigned __int64 i; // r12
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // rdx
  unsigned int v8; // eax
  __int64 v9; // rdi
  unsigned __int64 v10; // r12
  __int64 v11; // rax
  int v12; // edx
  unsigned __int64 v13; // r12
  unsigned int v14; // r15d
  unsigned __int64 v15; // r13
  int v16; // r14d
  char v17; // dl
  int v18; // r8d
  __int64 v19; // r9
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 *v22; // rax
  __int64 *v23; // rdi
  __int64 *v24; // rcx
  __int64 v26; // [rsp+18h] [rbp-138h]
  _QWORD *v27; // [rsp+20h] [rbp-130h] BYREF
  unsigned int v28; // [rsp+28h] [rbp-128h]
  unsigned int v29; // [rsp+2Ch] [rbp-124h]
  _QWORD v30[8]; // [rsp+30h] [rbp-120h] BYREF
  __int64 v31; // [rsp+70h] [rbp-E0h] BYREF
  __int64 *v32; // [rsp+78h] [rbp-D8h]
  __int64 *v33; // [rsp+80h] [rbp-D0h]
  unsigned int v34; // [rsp+88h] [rbp-C8h]
  unsigned int v35; // [rsp+8Ch] [rbp-C4h]
  int v36; // [rsp+90h] [rbp-C0h]
  _QWORD v37[23]; // [rsp+98h] [rbp-B8h] BYREF

  v1 = *(_DWORD *)(a1[3] + 8LL);
  v29 = 8;
  v34 = 16;
  v36 = 0;
  v2 = v1 >> 8 != 0;
  v3 = 0x9DDFEA08EB382D69LL
     * (a1[12]
      ^ (0x9DDFEA08EB382D69LL
       * (((0x9DDFEA08EB382D69LL
          * (v2
           ^ (0x9DDFEA08EB382D69LL * (v2 ^ 0x6ACAA36BEF8325C5LL))
           ^ ((0x9DDFEA08EB382D69LL * (v2 ^ 0x6ACAA36BEF8325C5LL)) >> 47))) >> 47)
        ^ (0x9DDFEA08EB382D69LL
         * (v2
          ^ (0x9DDFEA08EB382D69LL * (v2 ^ 0x6ACAA36BEF8325C5LL))
          ^ ((0x9DDFEA08EB382D69LL * (v2 ^ 0x6ACAA36BEF8325C5LL)) >> 47))))));
  i = 0x9DDFEA08EB382D69LL
    * (((0x9DDFEA08EB382D69LL * ((v3 >> 47) ^ a1[12] ^ v3)) >> 47) ^ (0x9DDFEA08EB382D69LL * ((v3 >> 47) ^ a1[12] ^ v3)));
  v27 = v30;
  v32 = v37;
  v33 = v37;
  v5 = a1[10];
  if ( v5 )
  {
    v5 -= 24;
    v6 = v5;
  }
  else
  {
    v6 = 0;
  }
  v28 = 1;
  v35 = 1;
  v31 = 1;
  v30[0] = v6;
  v7 = v30;
  v37[0] = v5;
  v8 = 1;
  while ( 1 )
  {
    v9 = v7[v8 - 1];
    v28 = v8 - 1;
    v10 = 0x9DDFEA08EB382D69LL
        * (((0x9DDFEA08EB382D69LL * (i ^ 0xB2E6)) >> 47) ^ (0x9DDFEA08EB382D69LL * (i ^ 0xB2E6)) ^ 0xB2E6);
    v11 = *(_QWORD *)(v9 + 48);
    for ( i = 0x9DDFEA08EB382D69LL * ((v10 >> 47) ^ v10); v9 + 40 != v11; i = 0x9DDFEA08EB382D69LL * ((v13 >> 47) ^ v13) )
    {
      if ( !v11 )
        BUG();
      v12 = *(unsigned __int8 *)(v11 - 8);
      v11 = *(_QWORD *)(v11 + 8);
      v13 = 0x9DDFEA08EB382D69LL
          * ((0x9DDFEA08EB382D69LL * ((unsigned int)(v12 - 24) ^ i))
           ^ (unsigned int)(v12 - 24)
           ^ ((0x9DDFEA08EB382D69LL * ((unsigned int)(v12 - 24) ^ i)) >> 47));
    }
    v14 = 0;
    v15 = sub_157EBA0(v9);
    v16 = sub_15F4D60(v15);
    if ( v16 )
    {
      while ( 1 )
      {
        v21 = sub_15F4DF0(v15, v14);
        v22 = v32;
        if ( v33 == v32 )
        {
          v23 = &v32[v35];
          if ( v32 != v23 )
          {
            v24 = 0;
            while ( v21 != *v22 )
            {
              if ( *v22 == -2 )
                v24 = v22;
              if ( v23 == ++v22 )
              {
                if ( !v24 )
                  goto LABEL_25;
                *v24 = v21;
                --v36;
                ++v31;
                goto LABEL_10;
              }
            }
            goto LABEL_13;
          }
LABEL_25:
          if ( v35 < v34 )
            break;
        }
        sub_16CCBA0((__int64)&v31, v21);
        if ( v17 )
          goto LABEL_10;
LABEL_13:
        if ( v16 == ++v14 )
          goto LABEL_23;
      }
      ++v35;
      *v23 = v21;
      ++v31;
LABEL_10:
      v19 = sub_15F4DF0(v15, v14);
      v20 = v28;
      if ( v28 >= v29 )
      {
        v26 = v19;
        sub_16CD150((__int64)&v27, v30, 0, 8, v18, v19);
        v20 = v28;
        v19 = v26;
      }
      v27[v20] = v19;
      ++v28;
      goto LABEL_13;
    }
LABEL_23:
    v8 = v28;
    if ( !v28 )
      break;
    v7 = v27;
  }
  if ( v33 != v32 )
    _libc_free((unsigned __int64)v33);
  if ( v27 != v30 )
    _libc_free((unsigned __int64)v27);
  return i;
}
