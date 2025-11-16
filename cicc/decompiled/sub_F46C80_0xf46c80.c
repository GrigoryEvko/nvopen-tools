// Function: sub_F46C80
// Address: 0xf46c80
//
_QWORD *__fastcall sub_F46C80(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  int v4; // edx
  __int64 v5; // rax
  _QWORD *v6; // r13
  int v7; // ecx
  __int64 v8; // rdx
  unsigned __int64 *v9; // r12
  __int64 v10; // rdx
  _QWORD *v11; // r12
  __int64 v12; // rdi
  unsigned int v13; // ecx
  _QWORD *v14; // r12
  __int64 v15; // rdx
  int v17; // ecx
  __int64 v18; // rdi
  _QWORD *v19; // r8
  int v20; // r9d
  unsigned int v21; // edx
  __int64 v22; // rsi
  int v23; // r9d
  int v24; // ecx
  int v25; // edx
  int v26; // ecx
  __int64 v27; // rdi
  int v28; // r9d
  unsigned int v29; // edx
  __int64 v30; // rsi
  _QWORD v31[2]; // [rsp+8h] [rbp-48h] BYREF
  __int64 v32; // [rsp+18h] [rbp-38h]
  __int64 v33; // [rsp+20h] [rbp-30h]

  v31[0] = 2;
  v31[1] = 0;
  v32 = a2;
  if ( a2 != -4096 && a2 != 0 && a2 != -8192 )
    sub_BD73F0((__int64)v31);
  v3 = *(_DWORD *)(a1 + 24);
  v33 = a1;
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
LABEL_6:
    sub_CF32C0(a1, 2 * v3);
    v4 = *(_DWORD *)(a1 + 24);
    if ( !v4 )
    {
LABEL_7:
      v5 = v32;
      v6 = 0;
LABEL_8:
      v7 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_9;
    }
    v5 = v32;
    v17 = v4 - 1;
    v18 = *(_QWORD *)(a1 + 8);
    v19 = 0;
    v20 = 1;
    v21 = (v4 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v6 = (_QWORD *)(v18 + ((unsigned __int64)v21 << 6));
    v22 = v6[3];
    if ( v32 == v22 )
      goto LABEL_8;
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v19 )
        v19 = v6;
      v21 = v17 & (v20 + v21);
      v6 = (_QWORD *)(v18 + ((unsigned __int64)v21 << 6));
      v22 = v6[3];
      if ( v32 == v22 )
        goto LABEL_8;
      ++v20;
    }
LABEL_27:
    if ( v19 )
      v6 = v19;
    goto LABEL_8;
  }
  v5 = v32;
  v12 = *(_QWORD *)(a1 + 8);
  v13 = (v3 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
  v14 = (_QWORD *)(v12 + ((unsigned __int64)v13 << 6));
  v15 = v14[3];
  if ( v15 == v32 )
  {
LABEL_20:
    v11 = v14 + 5;
    goto LABEL_21;
  }
  v23 = 1;
  v6 = 0;
  while ( v15 != -4096 )
  {
    if ( !v6 && v15 == -8192 )
      v6 = v14;
    v13 = (v3 - 1) & (v23 + v13);
    v14 = (_QWORD *)(v12 + ((unsigned __int64)v13 << 6));
    v15 = v14[3];
    if ( v32 == v15 )
      goto LABEL_20;
    ++v23;
  }
  v24 = *(_DWORD *)(a1 + 16);
  if ( !v6 )
    v6 = v14;
  ++*(_QWORD *)a1;
  v7 = v24 + 1;
  if ( 4 * v7 >= 3 * v3 )
    goto LABEL_6;
  if ( v3 - *(_DWORD *)(a1 + 20) - v7 <= v3 >> 3 )
  {
    sub_CF32C0(a1, v3);
    v25 = *(_DWORD *)(a1 + 24);
    if ( !v25 )
      goto LABEL_7;
    v5 = v32;
    v26 = v25 - 1;
    v27 = *(_QWORD *)(a1 + 8);
    v19 = 0;
    v28 = 1;
    v29 = (v25 - 1) & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v6 = (_QWORD *)(v27 + ((unsigned __int64)v29 << 6));
    v30 = v6[3];
    if ( v32 == v30 )
      goto LABEL_8;
    while ( v30 != -4096 )
    {
      if ( v30 == -8192 && !v19 )
        v19 = v6;
      v29 = v26 & (v28 + v29);
      v6 = (_QWORD *)(v27 + ((unsigned __int64)v29 << 6));
      v30 = v6[3];
      if ( v32 == v30 )
        goto LABEL_8;
      ++v28;
    }
    goto LABEL_27;
  }
LABEL_9:
  *(_DWORD *)(a1 + 16) = v7;
  if ( v6[3] == -4096 )
  {
    v9 = v6 + 1;
    if ( v5 != -4096 )
    {
LABEL_14:
      v6[3] = v5;
      if ( v5 != -4096 && v5 != 0 && v5 != -8192 )
        sub_BD6050(v9, v31[0] & 0xFFFFFFFFFFFFFFF8LL);
      v5 = v32;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 20);
    v8 = v6[3];
    if ( v5 != v8 )
    {
      v9 = v6 + 1;
      if ( v8 != -4096 && v8 != 0 && v8 != -8192 )
      {
        sub_BD60C0(v6 + 1);
        v5 = v32;
      }
      goto LABEL_14;
    }
  }
  v10 = v33;
  v6[5] = 6;
  v11 = v6 + 5;
  v6[6] = 0;
  v6[4] = v10;
  v6[7] = 0;
LABEL_21:
  if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
    sub_BD60C0(v31);
  return v11;
}
