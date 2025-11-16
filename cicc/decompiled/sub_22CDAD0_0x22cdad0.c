// Function: sub_22CDAD0
// Address: 0x22cdad0
//
void __fastcall sub_22CDAD0(unsigned __int64 a1)
{
  unsigned int v2; // ebx
  _BYTE *v3; // r9
  size_t v4; // rdx
  __int64 v5; // rax
  int v6; // ebx
  __int64 *v7; // rax
  __int64 v8; // r15
  unsigned __int8 *v9; // r13
  int v10; // r8d
  __int64 v11; // rsi
  int v12; // r8d
  int v13; // r9d
  unsigned int i; // edx
  _QWORD *v15; // rdi
  unsigned int v16; // edx
  unsigned __int64 v17; // rdi
  int v18; // eax
  __int64 *v19; // rax
  __int64 v20; // rsi
  bool v21; // zf
  int v22; // eax
  __int64 v23; // rdx
  unsigned int v24; // ecx
  _QWORD *v25; // rax
  _QWORD *j; // rdx
  unsigned int v27; // eax
  int v28; // eax
  unsigned __int64 v29; // rax
  unsigned __int64 v30; // rax
  int v31; // ebx
  __int64 v32; // r15
  unsigned __int8 v33[48]; // [rsp+10h] [rbp-F0h] BYREF
  _BYTE *v34; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v35; // [rsp+48h] [rbp-B8h]
  _BYTE v36[176]; // [rsp+50h] [rbp-B0h] BYREF

  v2 = *(_DWORD *)(a1 + 72);
  v34 = v36;
  v35 = 0x800000000LL;
  if ( !v2 )
    return;
  v3 = v36;
  v4 = 16LL * v2;
  if ( v2 <= 8
    || (sub_C8D5F0((__int64)&v34, v36, v2, 0x10u, v2, (__int64)v36),
        v3 = v34,
        v5 = *(unsigned int *)(a1 + 72),
        (v4 = 16 * v5) != 0) )
  {
    memcpy(v3, *(const void **)(a1 + 64), v4);
    v5 = *(unsigned int *)(a1 + 72);
  }
  LODWORD(v35) = v2;
  if ( !(_DWORD)v5 )
  {
LABEL_20:
    v17 = (unsigned __int64)v34;
    if ( v34 == v36 )
      return;
LABEL_21:
    _libc_free(v17);
    return;
  }
  v6 = 500;
  do
  {
    v7 = (__int64 *)(*(_QWORD *)(a1 + 64) + 16 * v5 - 16);
    v8 = *v7;
    v9 = (unsigned __int8 *)v7[1];
    if ( !(unsigned __int8)sub_22CDA20(a1, v9, *v7) )
    {
      v5 = *(unsigned int *)(a1 + 72);
LABEL_8:
      if ( !(_DWORD)v5 )
        goto LABEL_20;
      goto LABEL_9;
    }
    v10 = *(_DWORD *)(a1 + 232);
    v11 = *(_QWORD *)(a1 + 216);
    v5 = (unsigned int)(*(_DWORD *)(a1 + 72) - 1);
    *(_DWORD *)(a1 + 72) = v5;
    if ( !v10 )
      goto LABEL_8;
    v12 = v10 - 1;
    v13 = 1;
    for ( i = v12
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)
                | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
             ^ (484763065 * (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4)))); ; i = v12 & v16 )
    {
      v15 = (_QWORD *)(v11 + 16LL * i);
      if ( v8 == *v15 && v9 == (unsigned __int8 *)v15[1] )
        break;
      if ( *v15 == -4096 && v15[1] == -4096 )
        goto LABEL_8;
      v16 = v13 + i;
      ++v13;
    }
    *v15 = -8192;
    v15[1] = -8192;
    v5 = *(unsigned int *)(a1 + 72);
    --*(_DWORD *)(a1 + 224);
    ++*(_DWORD *)(a1 + 228);
    if ( !(_DWORD)v5 )
      goto LABEL_20;
LABEL_9:
    --v6;
  }
  while ( v6 );
  v18 = v35;
  if ( (_DWORD)v35 )
  {
    do
    {
      v19 = (__int64 *)&v34[16 * v18 - 16];
      v20 = v19[1];
      *(_WORD *)v33 = 6;
      sub_22C5240(a1, v20, *v19, (__int64)v33);
      sub_22C0090(v33);
      v21 = (_DWORD)v35 == 1;
      v18 = v35 - 1;
      LODWORD(v35) = v35 - 1;
    }
    while ( !v21 );
  }
  v22 = *(_DWORD *)(a1 + 224);
  ++*(_QWORD *)(a1 + 208);
  if ( v22 )
  {
    v24 = 4 * v22;
    v23 = *(unsigned int *)(a1 + 232);
    if ( (unsigned int)(4 * v22) < 0x40 )
      v24 = 64;
    if ( v24 >= (unsigned int)v23 )
      goto LABEL_35;
    v27 = v22 - 1;
    if ( v27 )
    {
      _BitScanReverse(&v27, v27);
      v28 = 1 << (33 - (v27 ^ 0x1F));
      if ( v28 < 64 )
        v28 = 64;
      if ( v28 == (_DWORD)v23 )
        goto LABEL_44;
      v29 = (4 * v28 / 3u + 1) | ((unsigned __int64)(4 * v28 / 3u + 1) >> 1);
      v30 = ((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2) | ((((v29 | (v29 >> 2)) >> 4) | v29 | (v29 >> 2)) >> 8);
      v31 = (v30 | (v30 >> 16)) + 1;
      v32 = 16 * ((v30 | (v30 >> 16)) + 1);
    }
    else
    {
      v32 = 2048;
      v31 = 128;
    }
    sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
    *(_DWORD *)(a1 + 232) = v31;
    *(_QWORD *)(a1 + 216) = sub_C7D670(v32, 8);
LABEL_44:
    sub_22C3CB0(a1 + 208);
    goto LABEL_28;
  }
  if ( !*(_DWORD *)(a1 + 228) )
    goto LABEL_28;
  v23 = *(unsigned int *)(a1 + 232);
  if ( (unsigned int)v23 > 0x40 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 216), 16LL * *(unsigned int *)(a1 + 232), 8);
    *(_QWORD *)(a1 + 216) = 0;
    *(_QWORD *)(a1 + 224) = 0;
    *(_DWORD *)(a1 + 232) = 0;
    goto LABEL_28;
  }
LABEL_35:
  v25 = *(_QWORD **)(a1 + 216);
  for ( j = &v25[2 * v23]; j != v25; *(v25 - 1) = -4096 )
  {
    *v25 = -4096;
    v25 += 2;
  }
  *(_QWORD *)(a1 + 224) = 0;
LABEL_28:
  *(_DWORD *)(a1 + 72) = 0;
  v17 = (unsigned __int64)v34;
  if ( v34 != v36 )
    goto LABEL_21;
}
