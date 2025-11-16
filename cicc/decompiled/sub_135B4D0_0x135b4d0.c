// Function: sub_135B4D0
// Address: 0x135b4d0
//
__int64 *__fastcall sub_135B4D0(__int64 a1, __int64 a2)
{
  unsigned int v3; // r9d
  __int64 v4; // rdi
  _QWORD *v5; // r12
  int v6; // edx
  __int64 v7; // r14
  unsigned int v8; // eax
  __int64 v9; // rdi
  __int64 v10; // rcx
  char v11; // r8
  __int64 v12; // rdx
  __int64 *result; // rax
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rcx
  unsigned int v17; // ecx
  __int64 v18; // r8
  __int64 v19; // rdi
  __int64 v20; // rdx
  _QWORD *v21; // r10
  int v22; // r11d
  unsigned int v23; // ecx
  __int64 v24; // r9
  __int64 v25; // rdi
  _QWORD *v26; // r9
  int v27; // r10d
  _QWORD *v28; // r10
  int v29; // r11d
  __int64 v30; // [rsp+8h] [rbp-D8h]
  __int64 v31; // [rsp+10h] [rbp-D0h]
  int v32; // [rsp+1Ch] [rbp-C4h]
  int v33; // [rsp+1Ch] [rbp-C4h]
  char v34; // [rsp+1Ch] [rbp-C4h]
  char v35; // [rsp+1Ch] [rbp-C4h]
  int v36; // [rsp+1Ch] [rbp-C4h]
  void *v37; // [rsp+20h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v39; // [rsp+38h] [rbp-A8h]
  __int64 v40; // [rsp+40h] [rbp-A0h]
  void *v41; // [rsp+50h] [rbp-90h] BYREF
  _BYTE v42[16]; // [rsp+58h] [rbp-88h] BYREF
  __int64 v43; // [rsp+68h] [rbp-78h]
  void *v44; // [rsp+80h] [rbp-60h] BYREF
  _BYTE v45[16]; // [rsp+88h] [rbp-58h] BYREF
  __int64 v46; // [rsp+98h] [rbp-48h]

  sub_1359800(&v37, a2, a1);
  v3 = *(_DWORD *)(a1 + 48);
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 24);
    v4 = a1 + 24;
    goto LABEL_3;
  }
  v7 = *(_QWORD *)(a1 + 32);
  v33 = *(_DWORD *)(a1 + 48);
  sub_1359800(&v41, -8, 0);
  sub_1359800(&v44, -16, 0);
  v8 = (v33 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
  v5 = (_QWORD *)(v7 + 48LL * v8);
  v9 = v5[3];
  if ( v39 == v9 )
  {
    v10 = v46;
    v11 = 1;
  }
  else
  {
    v10 = v46;
    v21 = (_QWORD *)(v7 + 48LL * ((v33 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))));
    v22 = 1;
    v5 = 0;
    while ( v43 != v9 )
    {
      if ( v46 != v9 || v5 )
        v21 = v5;
      v8 = (v33 - 1) & (v22 + v8);
      v5 = (_QWORD *)(v7 + 48LL * v8);
      v9 = v5[3];
      if ( v39 == v9 )
      {
        v11 = 1;
        goto LABEL_7;
      }
      ++v22;
      v5 = v21;
      v21 = (_QWORD *)(v7 + 48LL * v8);
    }
    v11 = 0;
    if ( !v5 )
      v5 = v21;
  }
LABEL_7:
  v44 = &unk_49EE2B0;
  if ( v10 != -8 && v10 != 0 && v10 != -16 )
  {
    v34 = v11;
    sub_1649B30(v45);
    v11 = v34;
  }
  v41 = &unk_49EE2B0;
  if ( v43 != -8 && v43 != 0 && v43 != -16 )
  {
    v35 = v11;
    sub_1649B30(v42);
    v11 = v35;
  }
  if ( v11 )
  {
    v12 = v39;
    goto LABEL_15;
  }
  v14 = *(_DWORD *)(a1 + 40);
  v3 = *(_DWORD *)(a1 + 48);
  v4 = a1 + 24;
  ++*(_QWORD *)(a1 + 24);
  v6 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v3 )
  {
LABEL_3:
    v5 = 0;
    sub_135AF50(v4, 2 * v3);
    v32 = *(_DWORD *)(a1 + 48);
    if ( !v32 )
      goto LABEL_4;
    v31 = *(_QWORD *)(a1 + 32);
    sub_1359800(&v41, -8, 0);
    sub_1359800(&v44, -16, 0);
    v17 = (v32 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v5 = (_QWORD *)(v31 + 48LL * v17);
    v18 = v5[3];
    if ( v39 == v18 )
    {
      v19 = v46;
    }
    else
    {
      v19 = v46;
      v26 = (_QWORD *)(v31 + 48LL * ((v32 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))));
      v27 = 1;
      v5 = 0;
      while ( v18 != v43 )
      {
        if ( !v5 && v18 == v46 )
          v5 = v26;
        v17 = (v32 - 1) & (v27 + v17);
        v26 = (_QWORD *)(v31 + 48LL * v17);
        v18 = v26[3];
        if ( v39 == v18 )
        {
          v5 = (_QWORD *)(v31 + 48LL * v17);
          goto LABEL_41;
        }
        ++v27;
      }
      if ( !v5 )
        v5 = v26;
    }
LABEL_41:
    v44 = &unk_49EE2B0;
    if ( v19 != 0 && v19 != -8 && v19 != -16 )
      sub_1649B30(v45);
    v41 = &unk_49EE2B0;
    v20 = v43;
    if ( v43 == 0 || v43 == -8 )
      goto LABEL_4;
    goto LABEL_45;
  }
  if ( v3 - (v6 + *(_DWORD *)(a1 + 44)) > v3 >> 3 )
    goto LABEL_22;
  v5 = 0;
  sub_135AF50(v4, v3);
  v36 = *(_DWORD *)(a1 + 48);
  if ( v36 )
  {
    v30 = *(_QWORD *)(a1 + 32);
    sub_1359800(&v41, -8, 0);
    sub_1359800(&v44, -16, 0);
    v23 = (v36 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4));
    v5 = (_QWORD *)(v30 + 48LL * v23);
    v24 = v5[3];
    if ( v39 == v24 )
    {
      v25 = v46;
    }
    else
    {
      v25 = v46;
      v28 = (_QWORD *)(v30 + 48LL * ((v36 - 1) & (((unsigned int)v39 >> 9) ^ ((unsigned int)v39 >> 4))));
      v29 = 1;
      v5 = 0;
      while ( v43 != v24 )
      {
        if ( !v5 && v46 == v24 )
          v5 = v28;
        v23 = (v36 - 1) & (v29 + v23);
        v28 = (_QWORD *)(v30 + 48LL * v23);
        v24 = v28[3];
        if ( v39 == v24 )
        {
          v5 = (_QWORD *)(v30 + 48LL * v23);
          goto LABEL_55;
        }
        ++v29;
      }
      if ( !v5 )
        v5 = v28;
    }
LABEL_55:
    v44 = &unk_49EE2B0;
    if ( v25 != 0 && v25 != -8 && v25 != -16 )
      sub_1649B30(v45);
    v41 = &unk_49EE2B0;
    v20 = v43;
    if ( v43 != -8 && v43 != 0 )
    {
LABEL_45:
      if ( v20 != -16 )
        sub_1649B30(v42);
    }
  }
LABEL_4:
  v6 = *(_DWORD *)(a1 + 40) + 1;
LABEL_22:
  *(_DWORD *)(a1 + 40) = v6;
  sub_1359800(&v44, -8, 0);
  v15 = v46;
  if ( v46 != v5[3] )
    --*(_DWORD *)(a1 + 44);
  v44 = &unk_49EE2B0;
  if ( v15 != -8 && v15 != 0 && v15 != -16 )
    sub_1649B30(v45);
  v16 = v5[3];
  v12 = v39;
  if ( v16 != v39 )
  {
    if ( v16 != -8 && v16 != 0 && v16 != -16 )
    {
      sub_1649B30(v5 + 1);
      v12 = v39;
    }
    v5[3] = v12;
    if ( v12 != 0 && v12 != -8 && v12 != -16 )
      sub_1649AC0(v5 + 1, v38 & 0xFFFFFFFFFFFFFFF8LL);
    v12 = v39;
  }
  v5[5] = 0;
  v5[4] = v40;
LABEL_15:
  v37 = &unk_49EE2B0;
  if ( v12 != 0 && v12 != -8 && v12 != -16 )
    sub_1649B30(&v38);
  result = (__int64 *)v5[5];
  if ( !result )
  {
    result = (__int64 *)sub_22077B0(64);
    if ( result )
    {
      *result = a2;
      result[1] = 0;
      result[2] = 0;
      result[3] = 0;
      result[4] = 0;
      result[5] = -8;
      result[6] = 0;
      result[7] = 0;
    }
    v5[5] = result;
  }
  return result;
}
