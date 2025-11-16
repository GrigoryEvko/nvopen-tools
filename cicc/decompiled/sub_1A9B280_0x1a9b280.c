// Function: sub_1A9B280
// Address: 0x1a9b280
//
__int64 __fastcall sub_1A9B280(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v3; // ebx
  char *v7; // rax
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdx
  char v12; // al
  __int64 *v13; // rdx
  unsigned int v14; // eax
  unsigned int v15; // esi
  unsigned int v16; // edi
  __int64 v17; // rax
  __int64 v18; // rdx
  char v19; // r9
  __int64 v20; // rdi
  __int64 *v21; // rsi
  _QWORD *v22; // rcx
  int v23; // r11d
  __int64 v24; // rsi
  unsigned int v25; // r12d
  __int64 *v26; // rax
  __int64 v27; // r13
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rax
  int v31; // eax
  int v32; // r14d
  __int64 v33; // rax
  __int64 v34; // [rsp+18h] [rbp-D8h]
  __int64 v35; // [rsp+20h] [rbp-D0h] BYREF
  __int64 *v36; // [rsp+28h] [rbp-C8h] BYREF
  __int64 v37; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v38; // [rsp+38h] [rbp-B8h]
  _QWORD *v39; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v40; // [rsp+48h] [rbp-A8h]
  char v41; // [rsp+C0h] [rbp-30h] BYREF

  v2 = 0;
  v3 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v3 != (*(_DWORD *)(a2 + 20) & 0xFFFFFFF) || *(_QWORD *)(a1 + 40) != *(_QWORD *)(a2 + 40) )
    return v2;
  v7 = (char *)&v39;
  v37 = 0;
  v38 = 1;
  do
  {
    *(_QWORD *)v7 = -8;
    v7 += 16;
  }
  while ( v7 != &v41 );
  if ( !v3 )
  {
    v2 = 1;
    v19 = v38 & 1;
    goto LABEL_45;
  }
  v8 = 8LL * v3;
  v9 = 0;
  v34 = v8;
  do
  {
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v11 = *(_QWORD *)(a1 - 8);
    else
      v11 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v35 = *(_QWORD *)(v11 + 3 * v9);
    v12 = sub_1A97280((__int64)&v37, &v35, &v36);
    v13 = v36;
    if ( v12 )
      goto LABEL_20;
    ++v37;
    v14 = ((unsigned int)v38 >> 1) + 1;
    if ( (v38 & 1) != 0 )
    {
      v16 = 24;
      v15 = 8;
    }
    else
    {
      v15 = v40;
      v16 = 3 * v40;
    }
    if ( 4 * v14 >= v16 )
    {
      v15 *= 2;
LABEL_39:
      sub_1A9AF30((__int64)&v37, v15);
      sub_1A97280((__int64)&v37, &v35, &v36);
      v13 = v36;
      v14 = ((unsigned int)v38 >> 1) + 1;
      goto LABEL_17;
    }
    if ( v15 - (v14 + HIDWORD(v38)) <= v15 >> 3 )
      goto LABEL_39;
LABEL_17:
    LODWORD(v38) = v38 & 1 | (2 * v14);
    if ( *v13 != -8 )
      --HIDWORD(v38);
    v17 = v35;
    v13[1] = 0;
    *v13 = v17;
LABEL_20:
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      v10 = *(_QWORD *)(a1 - 8);
    else
      v10 = a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF);
    v9 += 8;
    v13[1] = *(_QWORD *)(v9 + v10 + 24LL * *(unsigned int *)(a1 + 56));
  }
  while ( v34 != v9 );
  v18 = 0;
  v19 = v38 & 1;
  do
  {
    if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    {
      v20 = *(_QWORD *)(a2 - 8);
      v21 = (__int64 *)(v20 + 3 * v18);
      if ( !v19 )
        goto LABEL_35;
    }
    else
    {
      v20 = a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
      v21 = (__int64 *)(v20 + 3 * v18);
      if ( !v19 )
      {
LABEL_35:
        v29 = v40;
        v22 = v39;
        v23 = v40 - 1;
        if ( !v40 )
          goto LABEL_36;
        goto LABEL_27;
      }
    }
    v22 = &v39;
    v23 = 7;
LABEL_27:
    v24 = *v21;
    v25 = v23 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v26 = &v22[2 * v25];
    v27 = *v26;
    if ( *v26 == v24 )
      goto LABEL_28;
    v31 = 1;
    while ( v27 != -8 )
    {
      v32 = v31 + 1;
      v33 = v23 & (v25 + v31);
      v25 = v33;
      v26 = &v22[2 * v33];
      v27 = *v26;
      if ( v24 == *v26 )
        goto LABEL_28;
      v31 = v32;
    }
    if ( v19 )
    {
      v30 = 16;
      goto LABEL_37;
    }
    v29 = v40;
LABEL_36:
    v30 = 2 * v29;
LABEL_37:
    v26 = &v22[v30];
LABEL_28:
    v28 = 16;
    if ( !v19 )
      v28 = 2LL * v40;
    if ( v26 == &v22[v28] || v26[1] != *(_QWORD *)(v18 + v20 + 24LL * *(unsigned int *)(a2 + 56) + 8) )
    {
      v2 = 0;
      goto LABEL_45;
    }
    v18 += 8;
  }
  while ( v18 != v34 );
  v2 = 1;
LABEL_45:
  if ( !v19 )
    j___libc_free_0(v39);
  return v2;
}
