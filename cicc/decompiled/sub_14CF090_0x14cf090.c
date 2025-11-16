// Function: sub_14CF090
// Address: 0x14cf090
//
__int64 __fastcall sub_14CF090(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  unsigned int v8; // esi
  __int64 v9; // rdi
  int v10; // edx
  __int64 v11; // rax
  _QWORD *v12; // r12
  int v13; // ecx
  __int64 v14; // rdx
  _QWORD *v15; // r13
  __int64 v16; // rcx
  unsigned int v17; // esi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v21; // r8
  unsigned int v22; // edx
  __int64 v23; // rcx
  int v24; // ecx
  __int64 v25; // r9
  _QWORD *v26; // rdi
  int v27; // r8d
  unsigned int v28; // edx
  __int64 v29; // rsi
  int v30; // edx
  int v31; // r9d
  int v32; // r11d
  _QWORD *v33; // r10
  int v34; // ecx
  int v35; // edx
  int v36; // ecx
  __int64 v37; // r9
  int v38; // r8d
  unsigned int v39; // edx
  __int64 v40; // rsi
  _QWORD v41[2]; // [rsp+8h] [rbp-78h] BYREF
  __int64 v42; // [rsp+18h] [rbp-68h]
  __int64 v43; // [rsp+20h] [rbp-60h]
  void *v44; // [rsp+30h] [rbp-50h]
  __int64 v45; // [rsp+38h] [rbp-48h] BYREF
  __int64 v46; // [rsp+40h] [rbp-40h]
  __int64 v47; // [rsp+48h] [rbp-38h]
  __int64 v48; // [rsp+50h] [rbp-30h]
  __int64 v49; // [rsp+58h] [rbp-28h]

  v4 = *(unsigned int *)(a1 + 184);
  if ( (_DWORD)v4 )
  {
    v16 = *(_QWORD *)(a1 + 168);
    v17 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = v16 + 48LL * v17;
    v19 = *(_QWORD *)(v18 + 24);
    if ( a2 == v19 )
    {
LABEL_25:
      if ( v18 != 48 * v4 + v16 )
        return *(_QWORD *)(v18 + 40);
    }
    else
    {
      v30 = 1;
      while ( v19 != -8 )
      {
        v31 = v30 + 1;
        v17 = (v4 - 1) & (v30 + v17);
        v18 = v16 + 48LL * v17;
        v19 = *(_QWORD *)(v18 + 24);
        if ( a2 == v19 )
          goto LABEL_25;
        v30 = v31;
      }
    }
  }
  v5 = sub_22077B0(192);
  v6 = v5;
  if ( v5 )
  {
    *(_QWORD *)v5 = a2;
    *(_QWORD *)(v5 + 8) = v5 + 24;
    *(_QWORD *)(v5 + 16) = 0x400000000LL;
    *(_QWORD *)(v5 + 152) = 0;
    *(_QWORD *)(v5 + 160) = 0;
    *(_QWORD *)(v5 + 168) = 0;
    *(_DWORD *)(v5 + 176) = 0;
    *(_BYTE *)(v5 + 184) = 0;
  }
  v41[0] = 2;
  v41[1] = 0;
  v42 = a2;
  if ( a2 == -16 || a2 == -8 )
  {
    v43 = a1;
    v7 = a1;
    v45 = 2;
    v46 = 0;
    v47 = a2;
  }
  else
  {
    sub_164C220(v41);
    v43 = a1;
    v46 = 0;
    v47 = v42;
    v45 = v41[0] & 6;
    if ( v42 != -8 && v42 != -16 && v42 )
    {
      sub_1649AC0(&v45, v41[0] & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v43;
    }
    else
    {
      v7 = a1;
    }
  }
  v8 = *(_DWORD *)(a1 + 184);
  v48 = v7;
  v9 = a1 + 160;
  v44 = &unk_49ECBF8;
  v49 = v6;
  if ( !v8 )
  {
    ++*(_QWORD *)(a1 + 160);
LABEL_11:
    sub_14CE950(v9, 2 * v8);
    v10 = *(_DWORD *)(a1 + 184);
    if ( !v10 )
    {
LABEL_12:
      v11 = v47;
      v12 = 0;
LABEL_13:
      v13 = *(_DWORD *)(a1 + 176) + 1;
      goto LABEL_14;
    }
    v11 = v47;
    v24 = v10 - 1;
    v25 = *(_QWORD *)(a1 + 168);
    v26 = 0;
    v27 = 1;
    v28 = (v10 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v12 = (_QWORD *)(v25 + 48LL * v28);
    v29 = v12[3];
    if ( v29 == v47 )
      goto LABEL_13;
    while ( v29 != -8 )
    {
      if ( v29 == -16 && !v26 )
        v26 = v12;
      v28 = v24 & (v27 + v28);
      v12 = (_QWORD *)(v25 + 48LL * v28);
      v29 = v12[3];
      if ( v47 == v29 )
        goto LABEL_13;
      ++v27;
    }
LABEL_41:
    if ( v26 )
      v12 = v26;
    goto LABEL_13;
  }
  v11 = v47;
  v21 = *(_QWORD *)(a1 + 168);
  v22 = (v8 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
  v12 = (_QWORD *)(v21 + 48LL * v22);
  v23 = v12[3];
  if ( v47 == v23 )
  {
LABEL_28:
    if ( v6 )
    {
      sub_14CA8A0(v6);
      v11 = v47;
    }
    goto LABEL_30;
  }
  v32 = 1;
  v33 = 0;
  while ( v23 != -8 )
  {
    if ( v23 == -16 && !v33 )
      v33 = v12;
    v22 = (v8 - 1) & (v32 + v22);
    v12 = (_QWORD *)(v21 + 48LL * v22);
    v23 = v12[3];
    if ( v47 == v23 )
      goto LABEL_28;
    ++v32;
  }
  v34 = *(_DWORD *)(a1 + 176);
  if ( v33 )
    v12 = v33;
  ++*(_QWORD *)(a1 + 160);
  v13 = v34 + 1;
  if ( 4 * v13 >= 3 * v8 )
    goto LABEL_11;
  if ( v8 - *(_DWORD *)(a1 + 180) - v13 <= v8 >> 3 )
  {
    sub_14CE950(v9, v8);
    v35 = *(_DWORD *)(a1 + 184);
    if ( !v35 )
      goto LABEL_12;
    v11 = v47;
    v36 = v35 - 1;
    v37 = *(_QWORD *)(a1 + 168);
    v26 = 0;
    v38 = 1;
    v39 = (v35 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
    v12 = (_QWORD *)(v37 + 48LL * v39);
    v40 = v12[3];
    if ( v40 == v47 )
      goto LABEL_13;
    while ( v40 != -8 )
    {
      if ( !v26 && v40 == -16 )
        v26 = v12;
      v39 = v36 & (v38 + v39);
      v12 = (_QWORD *)(v37 + 48LL * v39);
      v40 = v12[3];
      if ( v47 == v40 )
        goto LABEL_13;
      ++v38;
    }
    goto LABEL_41;
  }
LABEL_14:
  *(_DWORD *)(a1 + 176) = v13;
  if ( v12[3] == -8 )
  {
    v15 = v12 + 1;
    if ( v11 != -8 )
    {
LABEL_19:
      v12[3] = v11;
      if ( v11 != 0 && v11 != -8 && v11 != -16 )
        sub_1649AC0(v15, v45 & 0xFFFFFFFFFFFFFFF8LL);
      v11 = v47;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 180);
    v14 = v12[3];
    if ( v11 != v14 )
    {
      v15 = v12 + 1;
      if ( v14 != -8 && v14 != 0 && v14 != -16 )
      {
        sub_1649B30(v12 + 1);
        v11 = v47;
      }
      goto LABEL_19;
    }
  }
  v12[4] = v48;
  v12[5] = v49;
LABEL_30:
  v44 = &unk_49EE2B0;
  if ( v11 != 0 && v11 != -8 && v11 != -16 )
    sub_1649B30(&v45);
  if ( v42 != 0 && v42 != -8 && v42 != -16 )
    sub_1649B30(v41);
  return v12[5];
}
