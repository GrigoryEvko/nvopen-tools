// Function: sub_2AD8840
// Address: 0x2ad8840
//
void __fastcall sub_2AD8840(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  _BYTE *v5; // rdi
  __int64 v6; // rbx
  __int64 v7; // r13
  __int64 *v8; // rdx
  _BYTE *v9; // rsi
  _BYTE *v10; // r14
  _BYTE *v11; // r13
  __int64 v12; // rsi
  char v13; // al
  _BYTE *v14; // r15
  __int64 v15; // r13
  __int64 v16; // r14
  int v17; // r12d
  unsigned int v18; // esi
  __int64 v19; // rdi
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // r9
  int v23; // r11d
  __int64 *v24; // r10
  int v25; // ecx
  int v26; // edx
  __int64 v27; // [rsp+10h] [rbp-F0h]
  __int64 v28; // [rsp+20h] [rbp-E0h]
  _BYTE *v29; // [rsp+28h] [rbp-D8h]
  __int64 *v30; // [rsp+38h] [rbp-C8h] BYREF
  __int64 v31; // [rsp+40h] [rbp-C0h] BYREF
  int v32; // [rsp+48h] [rbp-B8h]
  _BYTE *v33; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v34; // [rsp+58h] [rbp-A8h]
  _BYTE v35[48]; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v36; // [rsp+90h] [rbp-70h] BYREF
  char *v37; // [rsp+98h] [rbp-68h]
  __int64 v38; // [rsp+A0h] [rbp-60h]
  int v39; // [rsp+A8h] [rbp-58h]
  char v40; // [rsp+ACh] [rbp-54h]
  char v41; // [rsp+B0h] [rbp-50h] BYREF

  v33 = v35;
  v34 = 0x100000000LL;
  v4 = *(_QWORD *)(a1 + 32);
  v5 = v35;
  v6 = *(_QWORD *)(v4 + 112);
  v7 = v6 + 184LL * *(unsigned int *)(v4 + 120);
  if ( v6 == v7 )
    goto LABEL_24;
  do
  {
    v8 = *(__int64 **)(v6 + 40);
    v9 = *(_BYTE **)v6;
    v6 += 184;
    sub_2ABEEB0(a1, v9, v8, a2, (__int64)&v33);
  }
  while ( v7 != v6 );
  v10 = v33;
  v36 = 0;
  v38 = 4;
  v39 = 0;
  v5 = v33;
  v11 = &v33[40 * (unsigned int)v34];
  v40 = 1;
  v37 = &v41;
  if ( v33 == v11 )
    goto LABEL_24;
  do
  {
    v12 = *((_QWORD *)v10 + 3);
    v10 += 40;
    sub_AE6EC0((__int64)&v36, v12);
    v13 = v40;
  }
  while ( v10 != v11 );
  v29 = &v33[40 * (unsigned int)v34];
  if ( v29 == v33 )
    goto LABEL_22;
  v28 = a1;
  v14 = v33;
  v27 = a1 + 208;
  do
  {
    while ( 1 )
    {
      v15 = *(_QWORD *)v14;
      v16 = *((_QWORD *)v14 + 2);
      v17 = *((_DWORD *)v14 + 8);
      if ( sub_2AAA3F0(*(_QWORD *)(*((_QWORD *)v14 + 1) + 16LL), 0, (__int64)&v36)
        && sub_2AAA3F0(*(_QWORD *)(v16 + 16), 0, (__int64)&v36) )
      {
        v31 = v15;
        v32 = v17;
        v18 = *(_DWORD *)(v28 + 232);
        if ( !v18 )
        {
          v30 = 0;
          ++*(_QWORD *)(v28 + 208);
          goto LABEL_28;
        }
        v19 = *(_QWORD *)(v28 + 216);
        v20 = (v18 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v21 = (__int64 *)(v19 + 16LL * v20);
        v22 = *v21;
        if ( v15 != *v21 )
          break;
      }
LABEL_7:
      v14 += 40;
      if ( v29 == v14 )
        goto LABEL_21;
    }
    v23 = 1;
    v24 = 0;
    while ( v22 != -4096 )
    {
      if ( v22 == -8192 && !v24 )
        v24 = v21;
      v20 = (v18 - 1) & (v23 + v20);
      v21 = (__int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v15 == *v21 )
        goto LABEL_7;
      ++v23;
    }
    if ( v24 )
      v21 = v24;
    ++*(_QWORD *)(v28 + 208);
    v25 = *(_DWORD *)(v28 + 224);
    v30 = v21;
    v26 = v25 + 1;
    if ( 4 * (v25 + 1) < 3 * v18 )
    {
      if ( v18 - *(_DWORD *)(v28 + 228) - v26 > v18 >> 3 )
        goto LABEL_18;
      goto LABEL_29;
    }
LABEL_28:
    v18 *= 2;
LABEL_29:
    sub_A41E30(v27, v18);
    sub_2AC3AF0(v27, &v31, &v30);
    v15 = v31;
    v26 = *(_DWORD *)(v28 + 224) + 1;
    v21 = v30;
LABEL_18:
    *(_DWORD *)(v28 + 224) = v26;
    if ( *v21 != -4096 )
      --*(_DWORD *)(v28 + 228);
    *v21 = v15;
    v14 += 40;
    *((_DWORD *)v21 + 2) = v32;
  }
  while ( v29 != v14 );
LABEL_21:
  v13 = v40;
LABEL_22:
  v5 = v33;
  if ( !v13 )
  {
    _libc_free((unsigned __int64)v37);
    v5 = v33;
  }
LABEL_24:
  if ( v5 != v35 )
    _libc_free((unsigned __int64)v5);
}
