// Function: sub_1B06B50
// Address: 0x1b06b50
//
_QWORD *__fastcall sub_1B06B50(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  int v4; // eax
  __int64 v5; // rcx
  _QWORD *v6; // rbx
  int v7; // edx
  __int64 v8; // rax
  unsigned __int64 *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // edx
  __int64 v13; // rax
  int v15; // eax
  __int64 v16; // rdi
  _QWORD *v17; // r8
  int v18; // r9d
  unsigned int v19; // edx
  __int64 v20; // rsi
  int v21; // r10d
  _QWORD *v22; // r9
  int v23; // eax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  int v27; // r9d
  unsigned int v28; // edx
  __int64 v29; // rsi
  _QWORD v30[2]; // [rsp+8h] [rbp-38h] BYREF
  __int64 v31; // [rsp+18h] [rbp-28h]
  __int64 v32; // [rsp+20h] [rbp-20h]

  v30[0] = 2;
  v30[1] = 0;
  v31 = a2;
  if ( a2 != -8 && a2 != 0 && a2 != -16 )
    sub_164C220((__int64)v30);
  v3 = *(_DWORD *)(a1 + 24);
  v32 = a1;
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
LABEL_6:
    sub_12E48B0(a1, 2 * v3);
    v4 = *(_DWORD *)(a1 + 24);
    if ( !v4 )
    {
LABEL_7:
      v5 = v31;
      v6 = 0;
LABEL_8:
      v7 = *(_DWORD *)(a1 + 16) + 1;
      goto LABEL_9;
    }
    v5 = v31;
    v15 = v4 - 1;
    v16 = *(_QWORD *)(a1 + 8);
    v17 = 0;
    v18 = 1;
    v19 = v15 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v6 = (_QWORD *)(v16 + ((unsigned __int64)v19 << 6));
    v20 = v6[3];
    if ( v31 == v20 )
      goto LABEL_8;
    while ( v20 != -8 )
    {
      if ( v20 == -16 && !v17 )
        v17 = v6;
      v19 = v15 & (v18 + v19);
      v6 = (_QWORD *)(v16 + ((unsigned __int64)v19 << 6));
      v20 = v6[3];
      if ( v31 == v20 )
        goto LABEL_8;
      ++v18;
    }
LABEL_26:
    if ( v17 )
      v6 = v17;
    goto LABEL_8;
  }
  v5 = v31;
  v11 = *(_QWORD *)(a1 + 8);
  v12 = (v3 - 1) & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
  v6 = (_QWORD *)(v11 + ((unsigned __int64)v12 << 6));
  v13 = v6[3];
  if ( v13 == v31 )
    goto LABEL_20;
  v21 = 1;
  v22 = 0;
  while ( v13 != -8 )
  {
    if ( !v22 && v13 == -16 )
      v22 = v6;
    v12 = (v3 - 1) & (v21 + v12);
    v6 = (_QWORD *)(v11 + ((unsigned __int64)v12 << 6));
    v13 = v6[3];
    if ( v31 == v13 )
      goto LABEL_20;
    ++v21;
  }
  v23 = *(_DWORD *)(a1 + 16);
  if ( v22 )
    v6 = v22;
  ++*(_QWORD *)a1;
  v7 = v23 + 1;
  if ( 4 * (v23 + 1) >= 3 * v3 )
    goto LABEL_6;
  if ( v3 - *(_DWORD *)(a1 + 20) - v7 <= v3 >> 3 )
  {
    sub_12E48B0(a1, v3);
    v24 = *(_DWORD *)(a1 + 24);
    if ( !v24 )
      goto LABEL_7;
    v5 = v31;
    v25 = v24 - 1;
    v26 = *(_QWORD *)(a1 + 8);
    v17 = 0;
    v27 = 1;
    v28 = v25 & (((unsigned int)v31 >> 9) ^ ((unsigned int)v31 >> 4));
    v6 = (_QWORD *)(v26 + ((unsigned __int64)v28 << 6));
    v29 = v6[3];
    if ( v31 == v29 )
      goto LABEL_8;
    while ( v29 != -8 )
    {
      if ( v29 == -16 && !v17 )
        v17 = v6;
      v28 = v25 & (v27 + v28);
      v6 = (_QWORD *)(v26 + ((unsigned __int64)v28 << 6));
      v29 = v6[3];
      if ( v31 == v29 )
        goto LABEL_8;
      ++v27;
    }
    goto LABEL_26;
  }
LABEL_9:
  *(_DWORD *)(a1 + 16) = v7;
  if ( v6[3] == -8 )
  {
    v9 = v6 + 1;
    if ( v5 != -8 )
    {
LABEL_14:
      v6[3] = v5;
      if ( v5 != -8 && v5 != 0 && v5 != -16 )
        sub_1649AC0(v9, v30[0] & 0xFFFFFFFFFFFFFFF8LL);
      v5 = v31;
    }
  }
  else
  {
    --*(_DWORD *)(a1 + 20);
    v8 = v6[3];
    if ( v5 != v8 )
    {
      v9 = v6 + 1;
      if ( v8 != -8 && v8 != 0 && v8 != -16 )
      {
        sub_1649B30(v6 + 1);
        v5 = v31;
      }
      goto LABEL_14;
    }
  }
  v10 = v32;
  v6[5] = 6;
  v6[6] = 0;
  v6[4] = v10;
  v6[7] = 0;
LABEL_20:
  if ( v5 != 0 && v5 != -8 && v5 != -16 )
    sub_1649B30(v30);
  return v6 + 5;
}
