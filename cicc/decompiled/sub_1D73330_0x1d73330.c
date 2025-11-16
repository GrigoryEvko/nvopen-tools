// Function: sub_1D73330
// Address: 0x1d73330
//
void *__fastcall sub_1D73330(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r13
  unsigned __int64 v5; // rdi
  _QWORD *v6; // rax
  __int64 v7; // rcx
  _QWORD *v8; // rbx
  _QWORD *i; // rcx
  char v10; // dl
  _QWORD *v11; // r15
  __int64 j; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned __int64 *v17; // r8
  __int64 v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v22; // [rsp+8h] [rbp-B8h]
  unsigned __int64 *v23; // [rsp+10h] [rbp-B0h]
  _QWORD *v24; // [rsp+10h] [rbp-B0h]
  _QWORD *v25; // [rsp+28h] [rbp-98h] BYREF
  void *v26; // [rsp+30h] [rbp-90h]
  _QWORD v27[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v28; // [rsp+48h] [rbp-78h]
  __int64 v29; // [rsp+50h] [rbp-70h]
  void *v30; // [rsp+60h] [rbp-60h]
  _QWORD v31[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v32; // [rsp+78h] [rbp-48h]
  __int64 v33; // [rsp+80h] [rbp-40h]

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
  v6 = (_QWORD *)sub_22077B0((unsigned __int64)(unsigned int)v5 << 6);
  *(_QWORD *)(a1 + 8) = v6;
  if ( !v4 )
    return sub_1D671E0(a1);
  v7 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v31[0] = 2;
  v8 = &v4[8 * v3];
  v33 = 0;
  for ( i = &v6[8 * v7]; i != v6; v6 += 8 )
  {
    if ( v6 )
    {
      v10 = v31[0];
      v6[2] = 0;
      v6[3] = -8;
      *v6 = &unk_49F9E38;
      v6[1] = v10 & 6;
      v6[4] = v33;
    }
  }
  v27[1] = 0;
  v27[0] = 2;
  v28 = -8;
  v26 = &unk_49F9E38;
  v29 = 0;
  v31[0] = 2;
  v31[1] = 0;
  v32 = -16;
  v30 = &unk_49F9E38;
  v33 = 0;
  if ( v8 != v4 )
  {
    v11 = v4;
    for ( j = -8; ; j = v28 )
    {
      v13 = v11[3];
      if ( v13 != j )
      {
        j = v32;
        if ( v13 != v32 )
        {
          sub_1D682F0(a1, (__int64)v11, &v25);
          v14 = v25;
          v15 = v11[3];
          v16 = v25[3];
          if ( v16 != v15 )
          {
            v17 = v25 + 1;
            if ( v16 != -8 && v16 != 0 && v16 != -16 )
            {
              v22 = v25;
              v23 = v25 + 1;
              sub_1649B30(v25 + 1);
              v15 = v11[3];
              v14 = v22;
              v17 = v23;
            }
            v14[3] = v15;
            if ( v15 != -8 && v15 != 0 && v15 != -16 )
            {
              v24 = v14;
              sub_1649AC0(v17, v11[1] & 0xFFFFFFFFFFFFFFF8LL);
              v14 = v24;
            }
          }
          v18 = v11[4];
          v14[5] = 6;
          v14[6] = 0;
          v14[4] = v18;
          v19 = v11[7];
          v14[7] = v19;
          if ( v19 != 0 && v19 != -8 && v19 != -16 )
            sub_1649AC0(v14 + 5, v11[5] & 0xFFFFFFFFFFFFFFF8LL);
          ++*(_DWORD *)(a1 + 16);
          v20 = v11[7];
          if ( v20 != -8 && v20 != 0 && v20 != -16 )
            sub_1649B30(v11 + 5);
          j = v11[3];
        }
      }
      *v11 = &unk_49EE2B0;
      if ( j != 0 && j != -8 && j != -16 )
        sub_1649B30(v11 + 1);
      v11 += 8;
      if ( v8 == v11 )
        break;
    }
    v30 = &unk_49EE2B0;
    if ( v32 != -8 && v32 != 0 && v32 != -16 )
      sub_1649B30(v31);
  }
  v26 = &unk_49EE2B0;
  if ( v28 != 0 && v28 != -8 && v28 != -16 )
    sub_1649B30(v27);
  return (void *)j___libc_free_0(v4);
}
