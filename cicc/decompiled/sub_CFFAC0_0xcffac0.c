// Function: sub_CFFAC0
// Address: 0xcffac0
//
__int64 __fastcall sub_CFFAC0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rsi
  unsigned int v6; // ecx
  __int64 v7; // rdx
  __int64 v8; // r8
  int v10; // edx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // r14
  _QWORD *v14; // rax
  __int64 v15; // r13
  _QWORD *v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rsi
  __int64 v19; // rdi
  int v20; // edx
  __int64 v21; // rax
  _QWORD *v22; // r12
  int v23; // ecx
  __int64 v24; // rdx
  unsigned __int64 *v25; // r13
  __int64 v26; // r8
  unsigned int v27; // edx
  __int64 v28; // rcx
  int v29; // r9d
  int v30; // ecx
  __int64 v31; // r9
  _QWORD *v32; // rdi
  int v33; // r8d
  unsigned int v34; // edx
  __int64 v35; // rsi
  int v36; // r11d
  _QWORD *v37; // r10
  int v38; // ecx
  int v39; // edx
  int v40; // ecx
  __int64 v41; // r9
  int v42; // r8d
  unsigned int v43; // edx
  __int64 v44; // rsi
  _QWORD v45[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v46; // [rsp+18h] [rbp-68h]
  __int64 v47; // [rsp+20h] [rbp-60h]
  void *v48; // [rsp+30h] [rbp-50h]
  unsigned __int64 v49; // [rsp+38h] [rbp-48h] BYREF
  __int64 v50; // [rsp+40h] [rbp-40h]
  __int64 v51; // [rsp+48h] [rbp-38h]
  __int64 v52; // [rsp+50h] [rbp-30h]
  __int64 v53; // [rsp+58h] [rbp-28h]

  v4 = *(unsigned int *)(a1 + 200);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 184);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = v5 + 48LL * v6;
    v8 = *(_QWORD *)(v7 + 24);
    if ( a2 == v8 )
    {
LABEL_3:
      if ( v7 != v5 + 48 * v4 )
        return *(_QWORD *)(v7 + 40);
    }
    else
    {
      v10 = 1;
      while ( v8 != -4096 )
      {
        v29 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = v5 + 48LL * v6;
        v8 = *(_QWORD *)(v7 + 24);
        if ( a2 == v8 )
          goto LABEL_3;
        v10 = v29;
      }
    }
  }
  v11 = sub_B82360(*(_QWORD *)(a1 + 8), (__int64)&unk_4F89C28);
  if ( v11 && (v12 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v11 + 104LL))(v11, &unk_4F89C28)) != 0 )
    v13 = sub_DFED00(v12, a2);
  else
    v13 = 0;
  v14 = (_QWORD *)sub_22077B0(200);
  v15 = (__int64)v14;
  if ( v14 )
  {
    *v14 = a2;
    v16 = v14 + 4;
    *(v16 - 3) = v13;
    *(_QWORD *)(v15 + 16) = v16;
    *(_QWORD *)(v15 + 24) = 0x400000000LL;
    *(_QWORD *)(v15 + 160) = 0;
    *(_QWORD *)(v15 + 168) = 0;
    *(_QWORD *)(v15 + 176) = 0;
    *(_DWORD *)(v15 + 184) = 0;
    *(_BYTE *)(v15 + 192) = 0;
  }
  v45[0] = 2;
  v45[1] = 0;
  v46 = a2;
  if ( a2 == -4096 || a2 == -8192 )
  {
    v47 = a1;
    v17 = a1;
    v49 = 2;
    v50 = 0;
    v51 = a2;
  }
  else
  {
    sub_BD73F0((__int64)v45);
    v47 = a1;
    v50 = 0;
    v51 = v46;
    v49 = v45[0] & 6;
    if ( v46 != -8192 && v46 != -4096 && v46 )
    {
      sub_BD6050(&v49, v45[0] & 0xFFFFFFFFFFFFFFF8LL);
      v17 = v47;
    }
    else
    {
      v17 = a1;
    }
  }
  v18 = *(unsigned int *)(a1 + 200);
  v52 = v17;
  v19 = a1 + 176;
  v48 = &unk_49DDB10;
  v53 = v15;
  if ( !(_DWORD)v18 )
  {
    ++*(_QWORD *)(a1 + 176);
LABEL_19:
    sub_CFF330(v19, 2 * v18);
    v20 = *(_DWORD *)(a1 + 200);
    if ( !v20 )
    {
LABEL_20:
      v21 = v51;
      v22 = 0;
LABEL_21:
      v23 = *(_DWORD *)(a1 + 192) + 1;
      goto LABEL_22;
    }
    v21 = v51;
    v30 = v20 - 1;
    v31 = *(_QWORD *)(a1 + 184);
    v32 = 0;
    v33 = 1;
    v34 = (v20 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v22 = (_QWORD *)(v31 + 48LL * v34);
    v35 = v22[3];
    if ( v51 == v35 )
      goto LABEL_21;
    while ( v35 != -4096 )
    {
      if ( !v32 && v35 == -8192 )
        v32 = v22;
      v34 = v30 & (v33 + v34);
      v22 = (_QWORD *)(v31 + 48LL * v34);
      v35 = v22[3];
      if ( v51 == v35 )
        goto LABEL_21;
      ++v33;
    }
LABEL_49:
    if ( v32 )
      v22 = v32;
    goto LABEL_21;
  }
  v21 = v51;
  v26 = *(_QWORD *)(a1 + 184);
  v27 = (v18 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
  v22 = (_QWORD *)(v26 + 48LL * v27);
  v28 = v22[3];
  if ( v28 == v51 )
  {
LABEL_33:
    if ( v15 )
    {
      sub_CFB3D0(v15, v18);
      v21 = v51;
    }
    goto LABEL_35;
  }
  v36 = 1;
  v37 = 0;
  while ( v28 != -4096 )
  {
    if ( !v37 && v28 == -8192 )
      v37 = v22;
    v27 = (v18 - 1) & (v36 + v27);
    v22 = (_QWORD *)(v26 + 48LL * v27);
    v28 = v22[3];
    if ( v51 == v28 )
      goto LABEL_33;
    ++v36;
  }
  v38 = *(_DWORD *)(a1 + 192);
  if ( v37 )
    v22 = v37;
  ++*(_QWORD *)(a1 + 176);
  v23 = v38 + 1;
  if ( 4 * v23 >= (unsigned int)(3 * v18) )
    goto LABEL_19;
  if ( (int)v18 - *(_DWORD *)(a1 + 196) - v23 <= (unsigned int)v18 >> 3 )
  {
    sub_CFF330(v19, v18);
    v39 = *(_DWORD *)(a1 + 200);
    if ( !v39 )
      goto LABEL_20;
    v21 = v51;
    v40 = v39 - 1;
    v41 = *(_QWORD *)(a1 + 184);
    v32 = 0;
    v42 = 1;
    v43 = (v39 - 1) & (((unsigned int)v51 >> 9) ^ ((unsigned int)v51 >> 4));
    v22 = (_QWORD *)(v41 + 48LL * v43);
    v44 = v22[3];
    if ( v51 == v44 )
      goto LABEL_21;
    while ( v44 != -4096 )
    {
      if ( v44 == -8192 && !v32 )
        v32 = v22;
      v43 = v40 & (v42 + v43);
      v22 = (_QWORD *)(v41 + 48LL * v43);
      v44 = v22[3];
      if ( v51 == v44 )
        goto LABEL_21;
      ++v42;
    }
    goto LABEL_49;
  }
LABEL_22:
  *(_DWORD *)(a1 + 192) = v23;
  if ( v22[3] == -4096 )
  {
    v25 = v22 + 1;
    if ( v21 != -4096 )
    {
LABEL_27:
      v22[3] = v21;
      if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
        sub_BD6050(v25, v49 & 0xFFFFFFFFFFFFFFF8LL);
      v21 = v51;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 196);
    v24 = v22[3];
    if ( v21 != v24 )
    {
      v25 = v22 + 1;
      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
      {
        sub_BD60C0(v22 + 1);
        v21 = v51;
      }
      goto LABEL_27;
    }
  }
  v22[4] = v52;
  v22[5] = v53;
LABEL_35:
  v48 = &unk_49DB368;
  if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
    sub_BD60C0(&v49);
  if ( v46 != -4096 && v46 != 0 && v46 != -8192 )
    sub_BD60C0(v45);
  return v22[5];
}
