// Function: sub_3141C60
// Address: 0x3141c60
//
__int64 __fastcall sub_3141C60(__int64 a1, __int64 a2, _QWORD *a3, size_t a4, unsigned __int8 a5)
{
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r13
  __int64 v10; // r15
  unsigned int v11; // ecx
  int v12; // eax
  int v13; // eax
  __int64 v14; // r15
  int v16; // eax
  int v17; // r8d
  int v18; // ecx
  unsigned int i; // r10d
  __int64 v20; // r9
  const void *v21; // rsi
  int v22; // eax
  bool v23; // al
  unsigned int v24; // r10d
  int v25; // eax
  int v26; // r11d
  __int64 v27; // r10
  int v28; // ecx
  unsigned int k; // r9d
  const void *v30; // rsi
  unsigned int v31; // r9d
  int v32; // eax
  int v33; // r10d
  __int64 v34; // r9
  int v35; // r11d
  unsigned int j; // ecx
  const void *v37; // rsi
  unsigned int v38; // ecx
  int v39; // eax
  bool v40; // al
  int v41; // eax
  int v42; // [rsp+8h] [rbp-68h]
  int v43; // [rsp+8h] [rbp-68h]
  __int64 v44; // [rsp+10h] [rbp-60h]
  __int64 v45; // [rsp+10h] [rbp-60h]
  __int64 v46; // [rsp+10h] [rbp-60h]
  int v47; // [rsp+18h] [rbp-58h]
  unsigned int v48; // [rsp+18h] [rbp-58h]
  unsigned int v49; // [rsp+18h] [rbp-58h]
  unsigned int v50; // [rsp+1Ch] [rbp-54h]
  int v51; // [rsp+1Ch] [rbp-54h]
  int v52; // [rsp+1Ch] [rbp-54h]
  int v53; // [rsp+20h] [rbp-50h]
  int v54; // [rsp+20h] [rbp-50h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+28h] [rbp-48h]
  int v58; // [rsp+28h] [rbp-48h]
  int v59; // [rsp+28h] [rbp-48h]

  v7 = 32LL * *(unsigned int *)(a1 + 104) - 32;
  v8 = *(_QWORD *)(a1 + 96);
  if ( !a5 )
  {
    v9 = v7 + v8;
    v14 = a2 + 32;
    goto LABEL_10;
  }
  v9 = v7 + v8;
  v10 = 0;
  v11 = *(_DWORD *)(v9 + 24);
  if ( v11 )
  {
    v53 = *(_DWORD *)(v9 + 24);
    v57 = *(_QWORD *)(v9 + 8);
    v16 = sub_C94890(a3, a4);
    v17 = 1;
    v18 = v53 - 1;
    for ( i = (v53 - 1) & v16; ; i = v18 & v24 )
    {
      v20 = v57 + 48LL * i;
      v21 = *(const void **)v20;
      if ( *(_QWORD *)v20 == -1 )
      {
        v23 = (_QWORD *)((char *)a3 + 1) == 0;
      }
      else if ( v21 == (const void *)-2LL )
      {
        v23 = (_QWORD *)((char *)a3 + 2) == 0;
      }
      else
      {
        if ( *(_QWORD *)(v20 + 8) != a4 )
          goto LABEL_22;
        v44 = v57 + 48LL * i;
        v47 = v17;
        v50 = i;
        v54 = v18;
        if ( !a4 )
          goto LABEL_8;
        v22 = memcmp(a3, v21, a4);
        v18 = v54;
        i = v50;
        v17 = v47;
        v20 = v44;
        v23 = v22 == 0;
      }
      if ( v23 )
        goto LABEL_8;
      if ( v21 == (const void *)-1LL )
      {
        if ( !v10 )
          v10 = v20;
        v11 = *(_DWORD *)(v9 + 24);
        break;
      }
LABEL_22:
      if ( v10 || v21 != (const void *)-2LL )
        v20 = v10;
      v24 = v17 + i;
      v10 = v20;
      ++v17;
    }
  }
  v12 = *(_DWORD *)(v9 + 16);
  ++*(_QWORD *)v9;
  v13 = v12 + 1;
  if ( 4 * v13 < 3 * v11 )
  {
    if ( v11 - *(_DWORD *)(v9 + 20) - v13 > v11 >> 3 )
      goto LABEL_5;
    v10 = 0;
    sub_3141930(v9, v11);
    v59 = *(_DWORD *)(v9 + 24);
    if ( !v59 )
      goto LABEL_5;
    v56 = *(_QWORD *)(v9 + 8);
    v32 = sub_C94890(a3, a4);
    v33 = 1;
    v34 = 0;
    v35 = v59 - 1;
    for ( j = (v59 - 1) & v32; ; j = v35 & v38 )
    {
      v10 = v56 + 48LL * j;
      v37 = *(const void **)v10;
      if ( *(_QWORD *)v10 == -1 )
      {
        if ( a3 != (_QWORD *)-1LL && v34 )
          v10 = v34;
        goto LABEL_5;
      }
      if ( v37 == (const void *)-2LL )
      {
        if ( a3 == (_QWORD *)-2LL )
          goto LABEL_5;
      }
      else
      {
        if ( a4 != *(_QWORD *)(v10 + 8) )
          goto LABEL_38;
        v43 = v33;
        v46 = v34;
        v49 = j;
        v52 = v35;
        if ( !a4 )
          goto LABEL_5;
        v41 = memcmp(a3, v37, a4);
        v35 = v52;
        j = v49;
        v34 = v46;
        v33 = v43;
        if ( !v41 )
          goto LABEL_5;
      }
      if ( !v34 && v37 == (const void *)-2LL )
        v34 = v10;
LABEL_38:
      v38 = v33 + j;
      ++v33;
    }
  }
  v10 = 0;
  sub_3141930(v9, 2 * v11);
  v58 = *(_DWORD *)(v9 + 24);
  if ( !v58 )
    goto LABEL_5;
  v55 = *(_QWORD *)(v9 + 8);
  v25 = sub_C94890(a3, a4);
  v26 = 1;
  v27 = 0;
  v28 = v58 - 1;
  for ( k = (v58 - 1) & v25; ; k = v28 & v31 )
  {
    v10 = v55 + 48LL * k;
    v30 = *(const void **)v10;
    if ( *(_QWORD *)v10 == -1 )
      break;
    if ( v30 == (const void *)-2LL )
    {
      v40 = (_QWORD *)((char *)a3 + 2) == 0;
    }
    else
    {
      if ( *(_QWORD *)(v10 + 8) != a4 )
        goto LABEL_32;
      v42 = v26;
      v45 = v27;
      v48 = k;
      v51 = v28;
      if ( !a4 )
        goto LABEL_5;
      v39 = memcmp(a3, v30, a4);
      v28 = v51;
      k = v48;
      v27 = v45;
      v26 = v42;
      v40 = v39 == 0;
    }
    if ( v40 )
      goto LABEL_5;
    if ( v30 == (const void *)-2LL && !v27 )
      v27 = v10;
LABEL_32:
    v31 = v26 + k;
    ++v26;
  }
  if ( a3 != (_QWORD *)-1LL && v27 )
    v10 = v27;
LABEL_5:
  ++*(_DWORD *)(v9 + 16);
  if ( *(_QWORD *)v10 != -1 )
    --*(_DWORD *)(v9 + 20);
  *(_QWORD *)(v10 + 8) = a4;
  *(_QWORD *)(v10 + 16) = 1;
  *(_QWORD *)v10 = a3;
  *(_QWORD *)(v10 + 24) = 0;
  *(_QWORD *)(v10 + 32) = 0;
  *(_DWORD *)(v10 + 40) = 0;
LABEL_8:
  v14 = a2;
  sub_C7D6A0(0, 0, 8);
LABEL_10:
  sub_C7D6A0(*(_QWORD *)(v14 + 8), 24LL * *(unsigned int *)(v14 + 24), 8);
  ++*(_QWORD *)v14;
  *(_QWORD *)(v14 + 8) = 0;
  *(_QWORD *)(v14 + 16) = 0;
  *(_DWORD *)(v14 + 24) = 0;
  sub_C7D6A0(0, 0, 8);
  return (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD *, size_t, _QWORD))(*(_QWORD *)a1 + 24LL))(
           a1,
           v14,
           v9,
           a3,
           a4,
           a5);
}
