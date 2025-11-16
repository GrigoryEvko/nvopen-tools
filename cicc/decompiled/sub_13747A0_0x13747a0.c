// Function: sub_13747A0
// Address: 0x13747a0
//
unsigned __int64 __fastcall sub_13747A0(int *a1, __int64 a2)
{
  __int64 v2; // r8
  int v4; // eax
  unsigned int v5; // esi
  int v6; // ecx
  __int64 v7; // r10
  __int64 v8; // rdi
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r9
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rcx
  unsigned int v16; // r8d
  __int64 v17; // r9
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rcx
  unsigned __int64 result; // rax
  __int64 v22; // rsi
  int v23; // r13d
  __int64 *v24; // r12
  int v25; // ecx
  int v26; // ecx
  int v27; // eax
  int v28; // esi
  __int64 v29; // r9
  unsigned int v30; // edx
  __int64 v31; // r8
  int v32; // r11d
  __int64 *v33; // r10
  int v34; // eax
  int v35; // esi
  __int64 v36; // r9
  __int64 *v37; // r10
  int v38; // r11d
  unsigned int v39; // edx
  __int64 v40; // [rsp+8h] [rbp-58h] BYREF
  __int64 v41; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+20h] [rbp-40h]
  __int64 v44; // [rsp+28h] [rbp-38h]
  _QWORD *v45; // [rsp+30h] [rbp-30h]
  unsigned int v46; // [rsp+38h] [rbp-28h]

  v2 = (__int64)(a1 + 2);
  v4 = *a1;
  v40 = a2;
  v5 = a1[8];
  v6 = v4 + 1;
  *a1 = v4 + 1;
  if ( !v5 )
  {
    ++*((_QWORD *)a1 + 1);
    goto LABEL_29;
  }
  v7 = *((_QWORD *)a1 + 2);
  v8 = v40;
  v9 = (v5 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( v40 == *v10 )
    goto LABEL_3;
  v23 = 1;
  v24 = 0;
  while ( v11 != -8 )
  {
    if ( !v24 && v11 == -16 )
      v24 = v10;
    v9 = (v5 - 1) & (v23 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( v40 == *v10 )
      goto LABEL_3;
    ++v23;
  }
  v25 = a1[6];
  if ( v24 )
    v10 = v24;
  ++*((_QWORD *)a1 + 1);
  v26 = v25 + 1;
  if ( 4 * v26 >= 3 * v5 )
  {
LABEL_29:
    sub_13745E0(v2, 2 * v5);
    v27 = a1[8];
    if ( v27 )
    {
      v8 = v40;
      v28 = v27 - 1;
      v29 = *((_QWORD *)a1 + 2);
      v26 = a1[6] + 1;
      v30 = (v27 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v10 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v10;
      if ( *v10 != v40 )
      {
        v32 = 1;
        v33 = 0;
        while ( v31 != -8 )
        {
          if ( !v33 && v31 == -16 )
            v33 = v10;
          v30 = v28 & (v32 + v30);
          v10 = (__int64 *)(v29 + 16LL * v30);
          v31 = *v10;
          if ( v40 == *v10 )
            goto LABEL_25;
          ++v32;
        }
        if ( v33 )
          v10 = v33;
      }
      goto LABEL_25;
    }
    goto LABEL_57;
  }
  if ( v5 - a1[7] - v26 <= v5 >> 3 )
  {
    sub_13745E0(v2, v5);
    v34 = a1[8];
    if ( v34 )
    {
      v35 = v34 - 1;
      v36 = *((_QWORD *)a1 + 2);
      v37 = 0;
      v38 = 1;
      v26 = a1[6] + 1;
      v39 = (v34 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v10 = (__int64 *)(v36 + 16LL * v39);
      v8 = *v10;
      if ( v40 != *v10 )
      {
        while ( v8 != -8 )
        {
          if ( v8 == -16 && !v37 )
            v37 = v10;
          v39 = v35 & (v38 + v39);
          v10 = (__int64 *)(v36 + 16LL * v39);
          v8 = *v10;
          if ( v40 == *v10 )
            goto LABEL_25;
          ++v38;
        }
        v8 = v40;
        if ( v37 )
          v10 = v37;
      }
      goto LABEL_25;
    }
LABEL_57:
    ++a1[6];
    BUG();
  }
LABEL_25:
  a1[6] = v26;
  if ( *v10 != -8 )
    --a1[7];
  *v10 = v8;
  *((_DWORD *)v10 + 2) = 0;
  v6 = *a1;
LABEL_3:
  *((_DWORD *)v10 + 2) = v6;
  v12 = (_BYTE *)*((_QWORD *)a1 + 6);
  if ( v12 == *((_BYTE **)a1 + 7) )
  {
    sub_13725F0((__int64)(a1 + 10), v12, &v40);
    v13 = v40;
  }
  else
  {
    v13 = v40;
    if ( v12 )
    {
      *(_QWORD *)v12 = v40;
      v12 = (_BYTE *)*((_QWORD *)a1 + 6);
      v13 = v40;
    }
    *((_QWORD *)a1 + 6) = v12 + 8;
  }
  v14 = *(_QWORD *)(v13 + 32);
  v15 = *(unsigned int *)(v13 + 4);
  v16 = *a1;
  v17 = *(_QWORD *)(v13 + 40);
  v18 = *(_QWORD **)(v13 + 48);
  v19 = v15 + ((*(_QWORD *)(v13 + 24) - v14) >> 3);
  if ( v19 < 0 )
  {
    v20 = ~((unsigned __int64)~v19 >> 6);
    goto LABEL_10;
  }
  if ( v19 > 63 )
  {
    v20 = v19 >> 6;
LABEL_10:
    v18 += v20;
    v14 = *v18;
    v17 = *v18 + 512LL;
    result = *v18 + 8 * (v19 - (v20 << 6));
    goto LABEL_11;
  }
  result = *(_QWORD *)(v13 + 24) + 8 * v15;
LABEL_11:
  v43 = v14;
  v22 = *((_QWORD *)a1 + 12);
  v41 = v13;
  v42 = result;
  v44 = v17;
  v45 = v18;
  v46 = v16;
  if ( v22 == *((_QWORD *)a1 + 13) )
    return sub_1374390((__int64 *)a1 + 11, (char *)v22, (__int64)&v41);
  if ( v22 )
  {
    *(_QWORD *)v22 = v13;
    *(_QWORD *)(v22 + 8) = v42;
    *(_QWORD *)(v22 + 16) = v43;
    *(_QWORD *)(v22 + 24) = v44;
    *(_QWORD *)(v22 + 32) = v45;
    result = v46;
    *(_DWORD *)(v22 + 40) = v46;
    v22 = *((_QWORD *)a1 + 12);
  }
  *((_QWORD *)a1 + 12) = v22 + 48;
  return result;
}
