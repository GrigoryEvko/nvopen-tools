// Function: sub_1E105B0
// Address: 0x1e105b0
//
__int64 __fastcall sub_1E105B0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  int v10; // edx
  __int64 v11; // rax
  _QWORD *v12; // r15
  int v13; // ecx
  __int64 v14; // rdx
  unsigned __int64 *v15; // r13
  char v16; // cl
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // rcx
  unsigned int v20; // edx
  __int64 v21; // r8
  int v23; // ecx
  __int64 v24; // rdi
  _QWORD *v25; // r8
  int v26; // r9d
  unsigned int v27; // edx
  __int64 v28; // rsi
  int v29; // r11d
  _QWORD *v30; // r10
  int v31; // ecx
  int v32; // edx
  int v33; // edx
  __int64 v34; // rdi
  int v35; // r9d
  unsigned int v36; // ecx
  __int64 v37; // rsi
  char v38; // [rsp+Fh] [rbp-91h]
  char v39; // [rsp+Fh] [rbp-91h]
  _QWORD v40[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v41; // [rsp+28h] [rbp-78h]
  __int64 v42; // [rsp+30h] [rbp-70h]
  void *v43; // [rsp+40h] [rbp-60h]
  unsigned __int64 v44; // [rsp+48h] [rbp-58h] BYREF
  __int64 v45; // [rsp+50h] [rbp-50h]
  __int64 v46; // [rsp+58h] [rbp-48h]
  __int64 v47; // [rsp+60h] [rbp-40h]
  __int64 v48; // [rsp+68h] [rbp-38h]

  v6 = *a3;
  v40[1] = 0;
  v40[0] = 2;
  v41 = v6;
  if ( v6 == 0 || v6 == -8 || v6 == -16 )
  {
    v42 = a2;
    v44 = 2;
    v45 = 0;
    v46 = v6;
    v7 = a2;
  }
  else
  {
    sub_164C220((__int64)v40);
    v42 = a2;
    v45 = 0;
    v46 = v41;
    v44 = v40[0] & 6;
    if ( v41 == -8 || v41 == 0 || v41 == -16 )
    {
      v7 = a2;
    }
    else
    {
      sub_1649AC0(&v44, v40[0] & 0xFFFFFFFFFFFFFFF8LL);
      v7 = v42;
    }
  }
  v8 = a3[1];
  a3[1] = 0;
  v9 = *(unsigned int *)(a2 + 24);
  v43 = &unk_49FB768;
  v47 = v7;
  v48 = v8;
  if ( !(_DWORD)v9 )
  {
    ++*(_QWORD *)a2;
    goto LABEL_9;
  }
  v11 = v46;
  v19 = *(_QWORD *)(a2 + 8);
  v20 = (v9 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
  v12 = (_QWORD *)(v19 + 48LL * v20);
  v21 = v12[3];
  if ( v46 != v21 )
  {
    v29 = 1;
    v30 = 0;
    while ( v21 != -8 )
    {
      if ( !v30 && v21 == -16 )
        v30 = v12;
      v20 = (v9 - 1) & (v29 + v20);
      v12 = (_QWORD *)(v19 + 48LL * v20);
      v21 = v12[3];
      if ( v46 == v21 )
        goto LABEL_23;
      ++v29;
    }
    v31 = *(_DWORD *)(a2 + 16);
    if ( v30 )
      v12 = v30;
    ++*(_QWORD *)a2;
    v13 = v31 + 1;
    if ( 4 * v13 < (unsigned int)(3 * v9) )
    {
      if ( (int)v9 - *(_DWORD *)(a2 + 20) - v13 > (unsigned int)v9 >> 3 )
      {
LABEL_12:
        *(_DWORD *)(a2 + 16) = v13;
        if ( v12[3] == -8 )
        {
          v15 = v12 + 1;
          if ( v11 != -8 )
          {
LABEL_17:
            v12[3] = v11;
            if ( v11 != -8 && v11 != 0 && v11 != -16 )
              sub_1649AC0(v15, v44 & 0xFFFFFFFFFFFFFFF8LL);
            v11 = v46;
          }
        }
        else
        {
          --*(_DWORD *)(a2 + 20);
          v14 = v12[3];
          if ( v14 != v11 )
          {
            v15 = v12 + 1;
            if ( v14 != 0 && v14 != -8 && v14 != -16 )
            {
              sub_1649B30(v12 + 1);
              v11 = v46;
            }
            goto LABEL_17;
          }
        }
        v16 = 1;
        v12[4] = v47;
        v12[5] = v48;
        v17 = *(_QWORD *)a2;
        v18 = *(_QWORD *)(a2 + 8) + 48LL * *(unsigned int *)(a2 + 24);
        goto LABEL_26;
      }
      sub_1E101A0(a2, v9);
      v32 = *(_DWORD *)(a2 + 24);
      if ( !v32 )
        goto LABEL_10;
      v11 = v46;
      v33 = v32 - 1;
      v34 = *(_QWORD *)(a2 + 8);
      v25 = 0;
      v35 = 1;
      v36 = v33 & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
      v12 = (_QWORD *)(v34 + 48LL * v36);
      v37 = v12[3];
      if ( v37 == v46 )
        goto LABEL_11;
      while ( v37 != -8 )
      {
        if ( !v25 && v37 == -16 )
          v25 = v12;
        v36 = v33 & (v35 + v36);
        v12 = (_QWORD *)(v34 + 48LL * v36);
        v37 = v12[3];
        if ( v46 == v37 )
          goto LABEL_11;
        ++v35;
      }
      goto LABEL_35;
    }
LABEL_9:
    sub_1E101A0(a2, 2 * v9);
    v10 = *(_DWORD *)(a2 + 24);
    if ( !v10 )
    {
LABEL_10:
      v11 = v46;
      v12 = 0;
LABEL_11:
      v13 = *(_DWORD *)(a2 + 16) + 1;
      goto LABEL_12;
    }
    v11 = v46;
    v23 = v10 - 1;
    v24 = *(_QWORD *)(a2 + 8);
    v25 = 0;
    v26 = 1;
    v27 = (v10 - 1) & (((unsigned int)v46 >> 9) ^ ((unsigned int)v46 >> 4));
    v12 = (_QWORD *)(v24 + 48LL * v27);
    v28 = v12[3];
    if ( v46 == v28 )
      goto LABEL_11;
    while ( v28 != -8 )
    {
      if ( !v25 && v28 == -16 )
        v25 = v12;
      v27 = v23 & (v26 + v27);
      v12 = (_QWORD *)(v24 + 48LL * v27);
      v28 = v12[3];
      if ( v46 == v28 )
        goto LABEL_11;
      ++v26;
    }
LABEL_35:
    if ( v25 )
      v12 = v25;
    goto LABEL_11;
  }
LABEL_23:
  v17 = *(_QWORD *)a2;
  v18 = v19 + 48 * v9;
  if ( v8 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v8 + 16LL))(v8);
    v11 = v46;
  }
  v16 = 0;
LABEL_26:
  v43 = &unk_49EE2B0;
  if ( v11 != -8 && v11 != 0 && v11 != -16 )
  {
    v38 = v16;
    sub_1649B30(&v44);
    v16 = v38;
  }
  if ( v41 != -8 && v41 != 0 && v41 != -16 )
  {
    v39 = v16;
    sub_1649B30(v40);
    v16 = v39;
  }
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = v17;
  *(_QWORD *)(a1 + 16) = v12;
  *(_QWORD *)(a1 + 24) = v18;
  *(_BYTE *)(a1 + 32) = v16;
  return a1;
}
