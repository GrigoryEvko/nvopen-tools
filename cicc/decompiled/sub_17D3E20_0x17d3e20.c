// Function: sub_17D3E20
// Address: 0x17d3e20
//
__int64 __fastcall sub_17D3E20(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *v6; // rax
  __int64 v7; // rdx
  _QWORD *v8; // r12
  _QWORD *i; // rcx
  char v10; // dl
  _QWORD *v11; // rbx
  __int64 j; // rax
  __int64 v13; // rdx
  _QWORD *v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned __int64 *v17; // rdi
  _QWORD *v19; // [rsp+8h] [rbp-B8h]
  _QWORD *v20; // [rsp+10h] [rbp-B0h]
  _QWORD *v21; // [rsp+28h] [rbp-98h] BYREF
  void *v22; // [rsp+30h] [rbp-90h]
  _QWORD v23[2]; // [rsp+38h] [rbp-88h] BYREF
  __int64 v24; // [rsp+48h] [rbp-78h]
  __int64 v25; // [rsp+50h] [rbp-70h]
  void *v26; // [rsp+60h] [rbp-60h]
  _QWORD v27[2]; // [rsp+68h] [rbp-58h] BYREF
  __int64 v28; // [rsp+78h] [rbp-48h]
  __int64 v29; // [rsp+80h] [rbp-40h]

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
  v6 = (_QWORD *)sub_22077B0(48LL * (unsigned int)v5);
  *(_QWORD *)(a1 + 8) = v6;
  if ( !v4 )
    return sub_17D3930(a1);
  v7 = *(unsigned int *)(a1 + 24);
  *(_QWORD *)(a1 + 16) = 0;
  v27[0] = 2;
  v29 = 0;
  v8 = &v4[6 * v3];
  for ( i = &v6[6 * v7]; i != v6; v6 += 6 )
  {
    if ( v6 )
    {
      v10 = v27[0];
      v6[2] = 0;
      v6[3] = -8;
      *v6 = &unk_49F04B0;
      v6[1] = v10 & 6;
      v6[4] = v29;
    }
  }
  v23[1] = 0;
  v23[0] = 2;
  v24 = -8;
  v22 = &unk_49F04B0;
  v25 = 0;
  v27[0] = 2;
  v27[1] = 0;
  v28 = -16;
  v26 = &unk_49F04B0;
  v29 = 0;
  if ( v8 != v4 )
  {
    v11 = v4;
    for ( j = -8; ; j = v24 )
    {
      v13 = v11[3];
      if ( v13 != j )
      {
        j = v28;
        if ( v13 != v28 )
        {
          sub_17D3B80(a1, (__int64)v11, &v21);
          v14 = v21;
          v15 = v11[3];
          v16 = v21[3];
          if ( v16 != v15 )
          {
            v17 = v21 + 1;
            if ( v16 != -8 && v16 != 0 && v16 != -16 )
            {
              v19 = v21;
              sub_1649B30(v17);
              v15 = v11[3];
              v14 = v19;
            }
            v14[3] = v15;
            if ( v15 != 0 && v15 != -8 && v15 != -16 )
            {
              v20 = v14;
              sub_1649AC0(v17, v11[1] & 0xFFFFFFFFFFFFFFF8LL);
              v14 = v20;
            }
          }
          v14[4] = v11[4];
          v14[5] = v11[5];
          ++*(_DWORD *)(a1 + 16);
          j = v11[3];
        }
      }
      *v11 = &unk_49EE2B0;
      if ( j != -8 && j != 0 && j != -16 )
        sub_1649B30(v11 + 1);
      v11 += 6;
      if ( v8 == v11 )
        break;
    }
    v26 = &unk_49EE2B0;
    if ( v28 != 0 && v28 != -16 && v28 != -8 )
      sub_1649B30(v27);
  }
  v22 = &unk_49EE2B0;
  if ( v24 != -8 && v24 != 0 && v24 != -16 )
    sub_1649B30(v23);
  return j___libc_free_0(v4);
}
