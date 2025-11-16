// Function: sub_FEBEE0
// Address: 0xfebee0
//
unsigned __int64 __fastcall sub_FEBEE0(int *a1, __int64 a2)
{
  __int64 v2; // r9
  int v4; // eax
  unsigned int v5; // esi
  int v6; // r12d
  __int64 v7; // r8
  __int64 v8; // rdi
  __int64 *v9; // r11
  int v10; // r14d
  unsigned int v11; // ecx
  __int64 *v12; // rax
  __int64 v13; // rdx
  _BYTE *v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned int v18; // r8d
  __int64 v19; // r9
  _QWORD *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rsi
  unsigned __int64 result; // rax
  __int64 v24; // rsi
  int v25; // eax
  int v26; // edx
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  unsigned int v30; // ecx
  __int64 v31; // rax
  int v32; // r10d
  __int64 *v33; // r9
  int v34; // eax
  int v35; // eax
  __int64 v36; // r8
  int v37; // r10d
  unsigned int v38; // ecx
  __int64 v39; // rsi
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
    goto LABEL_33;
  }
  v7 = *((_QWORD *)a1 + 2);
  v8 = v40;
  v9 = 0;
  v10 = 1;
  v11 = (v5 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
  v12 = (__int64 *)(v7 + 16LL * v11);
  v13 = *v12;
  if ( v40 != *v12 )
  {
    while ( v13 != -4096 )
    {
      if ( !v9 && v13 == -8192 )
        v9 = v12;
      v11 = (v5 - 1) & (v10 + v11);
      v12 = (__int64 *)(v7 + 16LL * v11);
      v13 = *v12;
      if ( v40 == *v12 )
        goto LABEL_3;
      ++v10;
    }
    if ( !v9 )
      v9 = v12;
    v25 = a1[6];
    ++*((_QWORD *)a1 + 1);
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v5 )
    {
      if ( v5 - a1[7] - v26 > v5 >> 3 )
        goto LABEL_27;
      sub_FEBD00(v2, v5);
      v34 = a1[8];
      if ( v34 )
      {
        v8 = v40;
        v35 = v34 - 1;
        v36 = *((_QWORD *)a1 + 2);
        v33 = 0;
        v37 = 1;
        v38 = v35 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v26 = a1[6] + 1;
        v9 = (__int64 *)(v36 + 16LL * v38);
        v39 = *v9;
        if ( *v9 == v40 )
          goto LABEL_27;
        while ( v39 != -4096 )
        {
          if ( v39 == -8192 && !v33 )
            v33 = v9;
          v38 = v35 & (v37 + v38);
          v9 = (__int64 *)(v36 + 16LL * v38);
          v39 = *v9;
          if ( v40 == *v9 )
            goto LABEL_27;
          ++v37;
        }
LABEL_37:
        if ( v33 )
          v9 = v33;
LABEL_27:
        a1[6] = v26;
        if ( *v9 != -4096 )
          --a1[7];
        *v9 = v8;
        *((_DWORD *)v9 + 2) = 0;
        *((_DWORD *)v9 + 2) = v6;
        v14 = (_BYTE *)*((_QWORD *)a1 + 6);
        if ( v14 != *((_BYTE **)a1 + 7) )
          goto LABEL_4;
LABEL_30:
        sub_FE9C50((__int64)(a1 + 10), v14, &v40);
        v15 = v40;
        goto LABEL_7;
      }
      goto LABEL_53;
    }
LABEL_33:
    sub_FEBD00(v2, 2 * v5);
    v27 = a1[8];
    if ( v27 )
    {
      v8 = v40;
      v28 = v27 - 1;
      v29 = *((_QWORD *)a1 + 2);
      v26 = a1[6] + 1;
      v30 = (v27 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v9 = (__int64 *)(v29 + 16LL * v30);
      v31 = *v9;
      if ( *v9 == v40 )
        goto LABEL_27;
      v32 = 1;
      v33 = 0;
      while ( v31 != -4096 )
      {
        if ( !v33 && v31 == -8192 )
          v33 = v9;
        v30 = v28 & (v32 + v30);
        v9 = (__int64 *)(v29 + 16LL * v30);
        v31 = *v9;
        if ( v40 == *v9 )
          goto LABEL_27;
        ++v32;
      }
      goto LABEL_37;
    }
LABEL_53:
    ++a1[6];
    BUG();
  }
LABEL_3:
  *((_DWORD *)v12 + 2) = v6;
  v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  if ( v14 == *((_BYTE **)a1 + 7) )
    goto LABEL_30;
LABEL_4:
  v15 = v40;
  if ( v14 )
  {
    *(_QWORD *)v14 = v40;
    v14 = (_BYTE *)*((_QWORD *)a1 + 6);
  }
  *((_QWORD *)a1 + 6) = v14 + 8;
LABEL_7:
  v16 = *(_QWORD *)(v15 + 32);
  v17 = *(unsigned int *)(v15 + 4);
  v18 = *a1;
  v19 = *(_QWORD *)(v15 + 40);
  v20 = *(_QWORD **)(v15 + 48);
  v21 = v17 + ((*(_QWORD *)(v15 + 24) - v16) >> 3);
  if ( v21 < 0 )
  {
    v22 = ~((unsigned __int64)~v21 >> 6);
    goto LABEL_10;
  }
  if ( v21 > 63 )
  {
    v22 = v21 >> 6;
LABEL_10:
    v20 += v22;
    v16 = *v20;
    v19 = *v20 + 512LL;
    result = *v20 + 8 * (v21 - (v22 << 6));
    goto LABEL_11;
  }
  result = *(_QWORD *)(v15 + 24) + 8 * v17;
LABEL_11:
  v41 = v15;
  v24 = *((_QWORD *)a1 + 12);
  v42 = result;
  v43 = v16;
  v44 = v19;
  v45 = v20;
  v46 = v18;
  if ( v24 == *((_QWORD *)a1 + 13) )
    return sub_FEBAB0((__int64 *)a1 + 11, (char *)v24, (__int64)&v41);
  if ( v24 )
  {
    *(_QWORD *)v24 = v15;
    *(_QWORD *)(v24 + 8) = v42;
    *(_QWORD *)(v24 + 16) = v43;
    *(_QWORD *)(v24 + 24) = v44;
    *(_QWORD *)(v24 + 32) = v45;
    result = v46;
    *(_DWORD *)(v24 + 40) = v46;
    v24 = *((_QWORD *)a1 + 12);
  }
  *((_QWORD *)a1 + 12) = v24 + 48;
  return result;
}
