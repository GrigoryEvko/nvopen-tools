// Function: sub_F86EA0
// Address: 0xf86ea0
//
char __fastcall sub_F86EA0(__int64 a1, __int64 a2)
{
  bool v3; // al
  unsigned int v4; // esi
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  _QWORD *v8; // r13
  int v9; // edx
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // eax
  _QWORD *v13; // rcx
  __int64 v14; // rdx
  unsigned int v15; // esi
  _QWORD *v16; // r12
  int v17; // edx
  __int64 v18; // rbx
  __int64 v19; // r8
  unsigned int v20; // edx
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  int v23; // r9d
  int v24; // eax
  int v25; // r10d
  int v26; // eax
  _QWORD *v28; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v29[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v30; // [rsp+20h] [rbp-50h]
  __int64 v31; // [rsp+30h] [rbp-40h] BYREF
  __int64 v32; // [rsp+38h] [rbp-38h]
  __int64 v33; // [rsp+40h] [rbp-30h]

  v3 = a2 != -8192 && a2 != -4096 && a2 != 0;
  if ( *(_DWORD *)(a1 + 436) != *(_DWORD *)(a1 + 440) )
  {
    v31 = 0;
    v32 = 0;
    v33 = a2;
    if ( v3 )
    {
      v5 = a1 + 96;
      sub_BD73F0((__int64)&v31);
      v4 = *(_DWORD *)(a1 + 120);
      if ( !v4 )
        goto LABEL_4;
    }
    else
    {
      v4 = *(_DWORD *)(a1 + 120);
      v5 = a1 + 96;
      if ( !v4 )
      {
LABEL_4:
        ++*(_QWORD *)(a1 + 96);
        v29[0] = 0;
        goto LABEL_5;
      }
    }
    v7 = v33;
    v11 = *(_QWORD *)(a1 + 104);
    v12 = (v4 - 1) & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
    v13 = (_QWORD *)(v11 + 24LL * v12);
    v14 = v13[2];
    if ( v33 == v14 )
      goto LABEL_17;
    v23 = 1;
    v8 = 0;
    while ( v14 != -4096 )
    {
      if ( v8 || v14 != -8192 )
        v13 = v8;
      v12 = (v4 - 1) & (v23 + v12);
      v14 = *(_QWORD *)(v11 + 24LL * v12 + 16);
      if ( v33 == v14 )
        goto LABEL_17;
      v8 = v13;
      ++v23;
      v13 = (_QWORD *)(v11 + 24LL * v12);
    }
    v24 = *(_DWORD *)(a1 + 112);
    if ( !v8 )
      v8 = v13;
    ++*(_QWORD *)(a1 + 96);
    v9 = v24 + 1;
    v29[0] = v8;
    if ( 4 * (v24 + 1) < 3 * v4 )
    {
      LODWORD(v6) = v4 - *(_DWORD *)(a1 + 116) - v9;
      if ( (unsigned int)v6 > v4 >> 3 )
        goto LABEL_7;
      goto LABEL_6;
    }
LABEL_5:
    v4 *= 2;
LABEL_6:
    sub_F86930(v5, v4);
    sub_F82F60(v5, (__int64)&v31, v29);
    LODWORD(v6) = *(_DWORD *)(a1 + 112);
    v7 = v33;
    v8 = (_QWORD *)v29[0];
    v9 = v6 + 1;
LABEL_7:
    *(_DWORD *)(a1 + 112) = v9;
    if ( v8[2] == -4096 )
    {
      if ( v7 == -4096 )
        return v6;
      goto LABEL_12;
    }
    --*(_DWORD *)(a1 + 116);
    v10 = v8[2];
    if ( v10 != v7 )
    {
      if ( v10 != 0 && v10 != -4096 && v10 != -8192 )
        sub_BD60C0(v8);
LABEL_12:
      v8[2] = v7;
      if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
        sub_BD73F0((__int64)v8);
      v7 = v33;
    }
LABEL_17:
    LOBYTE(v6) = v7 != 0;
    if ( v7 != 0 && v7 != -4096 && v7 != -8192 )
      LOBYTE(v6) = sub_BD60C0(&v31);
    return v6;
  }
  v29[0] = 0;
  v29[1] = 0;
  v30 = a2;
  if ( v3 )
    sub_BD73F0((__int64)v29);
  v15 = *(_DWORD *)(a1 + 88);
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 64);
    v28 = 0;
    goto LABEL_27;
  }
  v6 = v30;
  v19 = *(_QWORD *)(a1 + 72);
  v20 = (v15 - 1) & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
  v21 = (_QWORD *)(v19 + 24LL * v20);
  v22 = v21[2];
  if ( v22 != v30 )
  {
    v25 = 1;
    v16 = 0;
    while ( v22 != -4096 )
    {
      if ( v16 || v22 != -8192 )
        v21 = v16;
      v20 = (v15 - 1) & (v25 + v20);
      v22 = *(_QWORD *)(v19 + 24LL * v20 + 16);
      if ( v30 == v22 )
        goto LABEL_40;
      ++v25;
      v16 = v21;
      v21 = (_QWORD *)(v19 + 24LL * v20);
    }
    v26 = *(_DWORD *)(a1 + 80);
    if ( !v16 )
      v16 = v21;
    ++*(_QWORD *)(a1 + 64);
    v17 = v26 + 1;
    v28 = v16;
    if ( 4 * (v26 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 84) - v17 > v15 >> 3 )
        goto LABEL_29;
      goto LABEL_28;
    }
LABEL_27:
    v15 *= 2;
LABEL_28:
    sub_F86930(a1 + 64, v15);
    sub_F82F60(a1 + 64, (__int64)v29, &v28);
    v16 = v28;
    v17 = *(_DWORD *)(a1 + 80) + 1;
LABEL_29:
    *(_DWORD *)(a1 + 80) = v17;
    v31 = 0;
    v32 = 0;
    v33 = -4096;
    if ( v16[2] != -4096 )
      --*(_DWORD *)(a1 + 84);
    sub_D68D70(&v31);
    v18 = v30;
    v6 = v16[2];
    if ( v30 != v6 )
    {
      if ( v6 != 0 && v6 != -4096 && v6 != -8192 )
        sub_BD60C0(v16);
      v16[2] = v18;
      if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        sub_BD73F0((__int64)v16);
      v6 = v30;
    }
  }
LABEL_40:
  if ( v6 != -4096 && v6 != 0 && v6 != -8192 )
    LOBYTE(v6) = sub_BD60C0(v29);
  return v6;
}
