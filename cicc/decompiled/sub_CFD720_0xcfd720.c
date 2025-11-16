// Function: sub_CFD720
// Address: 0xcfd720
//
_QWORD *__fastcall sub_CFD720(__int64 a1, __int64 a2)
{
  unsigned int v3; // r8d
  __int64 v4; // rcx
  unsigned int v5; // edx
  __int64 v6; // rax
  __int64 v7; // rdi
  _QWORD *result; // rax
  int v9; // eax
  int v10; // eax
  __int64 v11; // rsi
  _QWORD *v12; // r12
  int v13; // eax
  __int64 v14; // rax
  unsigned __int64 *v15; // r13
  unsigned int v16; // edi
  _QWORD *v17; // rax
  __int64 v18; // rdx
  int v19; // r10d
  int v20; // eax
  __int64 v21; // rdi
  int v22; // r9d
  unsigned int v23; // edx
  _QWORD *v24; // r8
  __int64 v25; // rcx
  int v26; // r11d
  int v27; // eax
  int v28; // eax
  int v29; // eax
  __int64 v30; // rdi
  int v31; // r9d
  unsigned int v32; // edx
  __int64 v33; // rcx
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  _QWORD v35[2]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v36; // [rsp+28h] [rbp-38h]
  __int64 v37; // [rsp+30h] [rbp-30h]

  v3 = *(_DWORD *)(a1 + 184);
  v4 = *(_QWORD *)(a1 + 168);
  if ( v3 )
  {
    v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v4 + 88LL * v5;
    v7 = *(_QWORD *)(v6 + 24);
    if ( v7 == a2 )
    {
LABEL_3:
      if ( v6 != v4 + 88LL * v3 )
        return (_QWORD *)(v6 + 40);
    }
    else
    {
      v9 = 1;
      while ( v7 != -4096 )
      {
        v19 = v9 + 1;
        v5 = (v3 - 1) & (v9 + v5);
        v6 = v4 + 88LL * v5;
        v7 = *(_QWORD *)(v6 + 24);
        if ( v7 == a2 )
          goto LABEL_3;
        v9 = v19;
      }
    }
  }
  v35[0] = 2;
  v35[1] = 0;
  v36 = a2;
  if ( a2 != -4096 && a2 != 0 && a2 != -8192 )
  {
    sub_BD73F0((__int64)v35);
    v4 = *(_QWORD *)(a1 + 168);
    v3 = *(_DWORD *)(a1 + 184);
  }
  v37 = a1;
  if ( !v3 )
  {
    ++*(_QWORD *)(a1 + 160);
    goto LABEL_13;
  }
  v11 = v36;
  v16 = (v3 - 1) & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
  v17 = (_QWORD *)(v4 + 88LL * v16);
  v18 = v17[3];
  if ( v36 != v18 )
  {
    v26 = 1;
    v12 = 0;
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v12 )
        v12 = v17;
      v16 = (v3 - 1) & (v26 + v16);
      v17 = (_QWORD *)(v4 + 88LL * v16);
      v18 = v17[3];
      if ( v36 == v18 )
        goto LABEL_27;
      ++v26;
    }
    if ( !v12 )
      v12 = v17;
    v27 = *(_DWORD *)(a1 + 176);
    ++*(_QWORD *)(a1 + 160);
    v13 = v27 + 1;
    if ( 4 * v13 < 3 * v3 )
    {
      if ( v3 - (v13 + *(_DWORD *)(a1 + 180)) > v3 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a1 + 176) = v13;
        if ( v12[3] == -4096 )
        {
          v15 = v12 + 1;
          if ( v11 != -4096 )
          {
LABEL_21:
            v12[3] = v11;
            if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
              sub_BD6050(v15, v35[0] & 0xFFFFFFFFFFFFFFF8LL);
            v11 = v36;
          }
        }
        else
        {
          --*(_DWORD *)(a1 + 180);
          v14 = v12[3];
          if ( v11 != v14 )
          {
            v15 = v12 + 1;
            if ( v14 != -4096 && v14 != 0 && v14 != -8192 )
            {
              sub_BD60C0(v12 + 1);
              v11 = v36;
            }
            goto LABEL_21;
          }
        }
        v12[4] = v37;
        v12[5] = v12 + 7;
        v12[6] = 0x100000000LL;
        result = v12 + 5;
        goto LABEL_28;
      }
      sub_CFD0B0(a1 + 160, v3);
      v28 = *(_DWORD *)(a1 + 184);
      if ( !v28 )
        goto LABEL_14;
      v11 = v36;
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 168);
      v31 = 1;
      v32 = v29 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
      v24 = 0;
      v12 = (_QWORD *)(v30 + 88LL * v32);
      v33 = v12[3];
      if ( v36 == v33 )
        goto LABEL_15;
      while ( v33 != -4096 )
      {
        if ( !v24 && v33 == -8192 )
          v24 = v12;
        v32 = v29 & (v31 + v32);
        v12 = (_QWORD *)(v30 + 88LL * v32);
        v33 = v12[3];
        if ( v36 == v33 )
          goto LABEL_15;
        ++v31;
      }
      goto LABEL_35;
    }
LABEL_13:
    sub_CFD0B0(a1 + 160, 2 * v3);
    v10 = *(_DWORD *)(a1 + 184);
    if ( !v10 )
    {
LABEL_14:
      v11 = v36;
      v12 = 0;
LABEL_15:
      v13 = *(_DWORD *)(a1 + 176) + 1;
      goto LABEL_16;
    }
    v11 = v36;
    v20 = v10 - 1;
    v21 = *(_QWORD *)(a1 + 168);
    v22 = 1;
    v23 = v20 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
    v24 = 0;
    v12 = (_QWORD *)(v21 + 88LL * v23);
    v25 = v12[3];
    if ( v25 == v36 )
      goto LABEL_15;
    while ( v25 != -4096 )
    {
      if ( !v24 && v25 == -8192 )
        v24 = v12;
      v23 = v20 & (v22 + v23);
      v12 = (_QWORD *)(v21 + 88LL * v23);
      v25 = v12[3];
      if ( v36 == v25 )
        goto LABEL_15;
      ++v22;
    }
LABEL_35:
    if ( v24 )
      v12 = v24;
    goto LABEL_15;
  }
LABEL_27:
  result = v17 + 5;
LABEL_28:
  if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
  {
    v34 = result;
    sub_BD60C0(v35);
    return v34;
  }
  return result;
}
