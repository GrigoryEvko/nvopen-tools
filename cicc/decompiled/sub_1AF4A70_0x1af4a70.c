// Function: sub_1AF4A70
// Address: 0x1af4a70
//
__int64 __fastcall sub_1AF4A70(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r15
  int v11; // eax
  __int64 v12; // rdx
  _QWORD *v13; // r13
  int v14; // ecx
  __int64 v15; // rax
  unsigned __int64 *v16; // r14
  __int64 v17; // rax
  __int64 v18; // rax
  bool v19; // zf
  unsigned __int64 v20; // r14
  __int64 v21; // r14
  unsigned int v22; // ecx
  __int64 v23; // rdi
  int v25; // ecx
  __int64 v26; // rdi
  _QWORD *v27; // r8
  int v28; // r9d
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // r10d
  _QWORD *v32; // r9
  int v33; // eax
  int v34; // eax
  int v35; // eax
  __int64 v36; // rdi
  int v37; // r9d
  unsigned int v38; // ecx
  __int64 v39; // rsi
  char v40; // [rsp+Fh] [rbp-A1h]
  _QWORD v41[2]; // [rsp+18h] [rbp-98h] BYREF
  __int64 v42; // [rsp+28h] [rbp-88h]
  __int64 v43; // [rsp+30h] [rbp-80h]
  void *v44; // [rsp+40h] [rbp-70h]
  unsigned __int64 v45; // [rsp+48h] [rbp-68h] BYREF
  __int64 v46; // [rsp+50h] [rbp-60h]
  __int64 v47; // [rsp+58h] [rbp-58h]
  __int64 v48; // [rsp+60h] [rbp-50h]
  unsigned __int64 v49[2]; // [rsp+68h] [rbp-48h] BYREF
  __int64 v50; // [rsp+78h] [rbp-38h]

  v6 = *a3;
  v41[0] = 2;
  v41[1] = 0;
  v42 = v6;
  if ( v6 == 0 || v6 == -8 || v6 == -16 )
  {
    v43 = a2;
    v45 = 2;
    v46 = 0;
    v47 = v6;
    v7 = a2;
  }
  else
  {
    sub_164C220((__int64)v41);
    v43 = a2;
    v46 = 0;
    v47 = v42;
    v45 = v41[0] & 6;
    if ( v42 == 0 || v42 == -8 || v42 == -16 )
    {
      v7 = a2;
    }
    else
    {
      sub_1649AC0(&v45, v41[0] & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v43;
    }
  }
  v48 = v7;
  v8 = a3[3];
  v44 = &unk_49E6B50;
  v49[0] = 6;
  v49[1] = 0;
  v50 = v8;
  if ( v8 != 0 && v8 != -8 && v8 != -16 )
    sub_1649AC0(v49, a3[1] & 0xFFFFFFFFFFFFFFF8LL);
  v9 = *(unsigned int *)(a2 + 24);
  v10 = *(_QWORD *)a2;
  if ( !(_DWORD)v9 )
  {
    *(_QWORD *)a2 = v10 + 1;
LABEL_12:
    sub_12E48B0(a2, 2 * v9);
    v11 = *(_DWORD *)(a2 + 24);
    if ( !v11 )
    {
LABEL_13:
      v12 = v47;
      v13 = 0;
LABEL_14:
      v14 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_15;
    }
    v12 = v47;
    v25 = v11 - 1;
    v26 = *(_QWORD *)(a2 + 8);
    v27 = 0;
    v28 = 1;
    v29 = (v11 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v13 = (_QWORD *)(v26 + ((unsigned __int64)v29 << 6));
    v30 = v13[3];
    if ( v47 == v30 )
      goto LABEL_14;
    while ( v30 != -8 )
    {
      if ( !v27 && v30 == -16 )
        v27 = v13;
      v29 = v25 & (v28 + v29);
      v13 = (_QWORD *)(v26 + ((unsigned __int64)v29 << 6));
      v30 = v13[3];
      if ( v47 == v30 )
        goto LABEL_14;
      ++v28;
    }
LABEL_41:
    if ( v27 )
      v13 = v27;
    goto LABEL_14;
  }
  v12 = v47;
  v21 = *(_QWORD *)(a2 + 8);
  v22 = (v9 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
  v13 = (_QWORD *)(v21 + ((unsigned __int64)v22 << 6));
  v23 = v13[3];
  if ( v47 == v23 )
  {
LABEL_28:
    v40 = 0;
    v20 = (v9 << 6) + v21;
    goto LABEL_29;
  }
  v31 = 1;
  v32 = 0;
  while ( v23 != -8 )
  {
    if ( v23 == -16 && !v32 )
      v32 = v13;
    v22 = (v9 - 1) & (v31 + v22);
    v13 = (_QWORD *)(v21 + ((unsigned __int64)v22 << 6));
    v23 = v13[3];
    if ( v47 == v23 )
      goto LABEL_28;
    ++v31;
  }
  *(_QWORD *)a2 = v10 + 1;
  v33 = *(_DWORD *)(a2 + 16);
  if ( v32 )
    v13 = v32;
  v14 = v33 + 1;
  if ( 4 * (v33 + 1) >= (unsigned int)(3 * v9) )
    goto LABEL_12;
  if ( (int)v9 - *(_DWORD *)(a2 + 20) - v14 <= (unsigned int)v9 >> 3 )
  {
    sub_12E48B0(a2, v9);
    v34 = *(_DWORD *)(a2 + 24);
    if ( !v34 )
      goto LABEL_13;
    v12 = v47;
    v35 = v34 - 1;
    v36 = *(_QWORD *)(a2 + 8);
    v27 = 0;
    v37 = 1;
    v38 = v35 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v13 = (_QWORD *)(v36 + ((unsigned __int64)v38 << 6));
    v39 = v13[3];
    if ( v47 == v39 )
      goto LABEL_14;
    while ( v39 != -8 )
    {
      if ( !v27 && v39 == -16 )
        v27 = v13;
      v38 = v35 & (v37 + v38);
      v13 = (_QWORD *)(v36 + ((unsigned __int64)v38 << 6));
      v39 = v13[3];
      if ( v47 == v39 )
        goto LABEL_14;
      ++v37;
    }
    goto LABEL_41;
  }
LABEL_15:
  *(_DWORD *)(a2 + 16) = v14;
  if ( v13[3] == -8 )
  {
    v16 = v13 + 1;
    if ( v12 != -8 )
    {
LABEL_20:
      v13[3] = v12;
      if ( v12 != -8 && v12 != 0 && v12 != -16 )
        sub_1649AC0(v16, v45 & 0xFFFFFFFFFFFFFFF8LL);
    }
  }
  else
  {
    --*(_DWORD *)(a2 + 20);
    v15 = v13[3];
    if ( v15 != v12 )
    {
      v16 = v13 + 1;
      if ( v15 != -8 && v15 != 0 && v15 != -16 )
      {
        sub_1649B30(v13 + 1);
        v12 = v47;
      }
      goto LABEL_20;
    }
  }
  v17 = v48;
  v13[5] = 6;
  v13[6] = 0;
  v13[4] = v17;
  v18 = v50;
  v19 = v50 == 0;
  v13[7] = v50;
  if ( v18 != -8 && !v19 && v18 != -16 )
    sub_1649AC0(v13 + 5, v49[0] & 0xFFFFFFFFFFFFFFF8LL);
  v10 = *(_QWORD *)a2;
  v40 = 1;
  v20 = *(_QWORD *)(a2 + 8) + ((unsigned __int64)*(unsigned int *)(a2 + 24) << 6);
LABEL_29:
  if ( v50 != 0 && v50 != -8 && v50 != -16 )
    sub_1649B30(v49);
  v44 = &unk_49EE2B0;
  if ( v47 != -8 && v47 != 0 && v47 != -16 )
    sub_1649B30(&v45);
  if ( v42 != -8 && v42 != 0 && v42 != -16 )
    sub_1649B30(v41);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v10;
  *(_QWORD *)(a1 + 16) = v13;
  *(_QWORD *)(a1 + 24) = v20;
  *(_BYTE *)(a1 + 32) = v40;
  return a1;
}
