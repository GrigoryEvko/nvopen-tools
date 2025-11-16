// Function: sub_14676C0
// Address: 0x14676c0
//
__int64 __fastcall sub_14676C0(__int64 a1, int a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // r14
  unsigned __int64 v5; // rax
  _QWORD *v6; // r15
  _QWORD *v7; // rbx
  _QWORD *i; // r13
  char v9; // al
  __int64 v10; // rax
  bool v11; // zf
  _QWORD *j; // rbx
  __int64 v13; // rcx
  __int64 v14; // rdx
  int v15; // r13d
  __int64 v16; // rdi
  int v17; // ecx
  unsigned int v18; // esi
  __int64 v19; // rdx
  _QWORD *v20; // r13
  __int64 v21; // r10
  __int64 v22; // r9
  __int64 v23; // rsi
  __int64 v24; // rcx
  _QWORD *v25; // r8
  __int64 result; // rax
  _QWORD *v27; // rbx
  _QWORD *k; // r12
  char v29; // al
  __int64 v30; // rax
  _QWORD *v31; // rax
  int v32; // edx
  _QWORD *v33; // r11
  __int64 v34; // [rsp+10h] [rbp-100h]
  void *v35; // [rsp+20h] [rbp-F0h] BYREF
  char v36[16]; // [rsp+28h] [rbp-E8h] BYREF
  __int64 v37; // [rsp+38h] [rbp-D8h]
  void *v38; // [rsp+50h] [rbp-C0h] BYREF
  char v39[16]; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v40; // [rsp+68h] [rbp-A8h]
  void *v41; // [rsp+80h] [rbp-90h] BYREF
  char v42[16]; // [rsp+88h] [rbp-88h] BYREF
  __int64 v43; // [rsp+98h] [rbp-78h]
  void *v44; // [rsp+B0h] [rbp-60h] BYREF
  _QWORD v45[2]; // [rsp+B8h] [rbp-58h] BYREF
  __int64 v46; // [rsp+C8h] [rbp-48h]
  __int64 v47; // [rsp+D0h] [rbp-40h]

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
  *(_QWORD *)(a1 + 8) = sub_22077B0(48LL * (unsigned int)v5);
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    v6 = &v4[6 * v3];
    sub_1457D90(&v44, -8, 0);
    v7 = *(_QWORD **)(a1 + 8);
    for ( i = &v7[6 * *(unsigned int *)(a1 + 24)]; i != v7; v7 += 6 )
    {
      if ( v7 )
      {
        v9 = v45[0];
        v7[2] = 0;
        v7[1] = v9 & 6;
        v10 = v46;
        v11 = v46 == -8;
        v7[3] = v46;
        if ( v10 != 0 && !v11 && v10 != -16 )
          sub_1649AC0(v7 + 1, v45[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v7 = &unk_49EC5C8;
        v7[4] = v47;
      }
    }
    v44 = &unk_49EE2B0;
    if ( v46 != 0 && v46 != -8 && v46 != -16 )
      sub_1649B30(v45);
    sub_1457D90(&v35, -8, 0);
    sub_1457D90(&v38, -16, 0);
    for ( j = v4; v6 != j; j += 6 )
    {
      v13 = j[3];
      if ( v37 == v13 )
      {
        v14 = j[3];
      }
      else
      {
        v14 = v40;
        if ( v13 != v40 )
        {
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
            BUG();
          v34 = *(_QWORD *)(a1 + 8);
          sub_1457D90(&v41, -8, 0);
          sub_1457D90(&v44, -16, 0);
          v16 = j[3];
          v17 = v15 - 1;
          v18 = (v15 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v19 = 48LL * v18;
          v20 = (_QWORD *)(v34 + v19);
          v21 = *(_QWORD *)(v34 + v19 + 24);
          if ( v21 == v16 )
          {
            v22 = v46;
          }
          else
          {
            v22 = v46;
            v31 = (_QWORD *)(v34 + v19);
            v32 = 1;
            v20 = 0;
            while ( v43 != v21 )
            {
              if ( v46 != v21 || v20 )
                v31 = v20;
              v18 = v17 & (v32 + v18);
              v20 = (_QWORD *)(v34 + 48LL * v18);
              v21 = v20[3];
              if ( v16 == v21 )
                goto LABEL_20;
              ++v32;
              v33 = v31;
              v31 = (_QWORD *)(v34 + 48LL * v18);
              v20 = v33;
            }
            if ( !v20 )
              v20 = v31;
          }
LABEL_20:
          v44 = &unk_49EE2B0;
          if ( v22 != -8 && v22 != 0 && v22 != -16 )
            sub_1649B30(v45);
          v41 = &unk_49EE2B0;
          if ( v43 != -8 && v43 != 0 && v43 != -16 )
            sub_1649B30(v42);
          v23 = v20[3];
          v24 = j[3];
          v25 = v20 + 1;
          if ( v23 != v24 )
          {
            if ( v23 != 0 && v23 != -8 && v23 != -16 )
            {
              sub_1649B30(v20 + 1);
              v24 = j[3];
              v25 = v20 + 1;
            }
            v20[3] = v24;
            if ( v24 != -8 && v24 != 0 && v24 != -16 )
              sub_1649AC0(v25, j[1] & 0xFFFFFFFFFFFFFFF8LL);
          }
          v20[4] = j[4];
          v20[5] = j[5];
          ++*(_DWORD *)(a1 + 16);
          v14 = j[3];
        }
      }
      *j = &unk_49EE2B0;
      if ( v14 != 0 && v14 != -8 && v14 != -16 )
        sub_1649B30(j + 1);
    }
    v38 = &unk_49EE2B0;
    if ( v40 != 0 && v40 != -8 && v40 != -16 )
      sub_1649B30(v39);
    v35 = &unk_49EE2B0;
    if ( v37 != -8 && v37 != 0 && v37 != -16 )
      sub_1649B30(v36);
    return j___libc_free_0(v4);
  }
  else
  {
    *(_QWORD *)(a1 + 16) = 0;
    sub_1457D90(&v44, -8, 0);
    v27 = *(_QWORD **)(a1 + 8);
    for ( k = &v27[6 * *(unsigned int *)(a1 + 24)]; k != v27; v27 += 6 )
    {
      if ( v27 )
      {
        v29 = v45[0];
        v27[2] = 0;
        v27[1] = v29 & 6;
        v30 = v46;
        v11 = v46 == 0;
        v27[3] = v46;
        if ( v30 != -8 && !v11 && v30 != -16 )
          sub_1649AC0(v27 + 1, v45[0] & 0xFFFFFFFFFFFFFFF8LL);
        *v27 = &unk_49EC5C8;
        v27[4] = v47;
      }
    }
    v44 = &unk_49EE2B0;
    result = v46;
    if ( v46 != 0 && v46 != -8 && v46 != -16 )
      return sub_1649B30(v45);
  }
  return result;
}
