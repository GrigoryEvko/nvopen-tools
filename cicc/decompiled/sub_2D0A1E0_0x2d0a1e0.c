// Function: sub_2D0A1E0
// Address: 0x2d0a1e0
//
void __fastcall sub_2D0A1E0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v4; // rax
  __int64 v5; // rbx
  char *v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  char *v10; // rax
  char *v11; // r14
  __int64 v12; // rbx
  char *v13; // r12
  unsigned __int64 v14; // rax
  char *v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rcx
  char *v18; // rax
  char *v19; // rsi
  char v20; // dl
  unsigned __int8 *v21; // rax
  unsigned __int8 v22; // dl
  __int64 v23; // rax
  unsigned int v24; // esi
  __int64 v25; // rdi
  int v26; // r11d
  __int64 *v27; // rdx
  __int64 v28; // r9
  unsigned int v29; // ecx
  _QWORD *v30; // rax
  __int64 v31; // r8
  __int64 v32; // r15
  int v33; // eax
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // r14
  __int64 v37; // rax
  char *v38; // rsi
  int v39; // eax
  int v40; // ecx
  int *v41; // [rsp+8h] [rbp-168h]
  __int64 v42; // [rsp+10h] [rbp-160h] BYREF
  __int64 *v43; // [rsp+18h] [rbp-158h] BYREF
  _QWORD v44[2]; // [rsp+20h] [rbp-150h] BYREF
  int v45; // [rsp+30h] [rbp-140h]
  char v46[8]; // [rsp+38h] [rbp-138h] BYREF
  unsigned __int64 v47; // [rsp+40h] [rbp-130h]
  char v48; // [rsp+54h] [rbp-11Ch]
  char v49[72]; // [rsp+58h] [rbp-118h] BYREF
  __int64 v50; // [rsp+A0h] [rbp-D0h] BYREF
  char *v51; // [rsp+A8h] [rbp-C8h]
  __int64 v52; // [rsp+B0h] [rbp-C0h]
  int v53; // [rsp+B8h] [rbp-B8h]
  char v54; // [rsp+BCh] [rbp-B4h]
  char v55; // [rsp+C0h] [rbp-B0h] BYREF

  v51 = &v55;
  v4 = *(unsigned __int8 **)a2;
  v52 = 16;
  v53 = 0;
  v54 = 1;
  v5 = *((_QWORD *)v4 + 2);
  v50 = 0;
  if ( v5 )
  {
    while ( 1 )
    {
      v42 = sub_2D05370(a1, v5, 1);
      if ( !v42 )
        goto LABEL_8;
      if ( !v54 )
        goto LABEL_17;
      v10 = v51;
      v7 = HIDWORD(v52);
      v6 = &v51[8 * HIDWORD(v52)];
      if ( v51 != v6 )
      {
        while ( v42 != *(_QWORD *)v10 )
        {
          v10 += 8;
          if ( v6 == v10 )
            goto LABEL_30;
        }
        goto LABEL_8;
      }
LABEL_30:
      if ( HIDWORD(v52) < (unsigned int)v52 )
      {
        ++HIDWORD(v52);
        *(_QWORD *)v6 = v42;
        v21 = *(unsigned __int8 **)a2;
        ++v50;
        v22 = *v21;
        if ( *v21 > 0x1Cu )
        {
LABEL_19:
          v23 = *((_QWORD *)v21 + 5);
          goto LABEL_20;
        }
LABEL_32:
        if ( v22 != 22 )
          BUG();
        v23 = *(_QWORD *)(*((_QWORD *)v21 + 3) + 80LL);
        if ( v23 )
          v23 -= 24;
LABEL_20:
        if ( v42 == v23 )
          goto LABEL_8;
        sub_2D08230((__int64)v44, a1, a2, v42, 1);
        v24 = *(_DWORD *)(a1 + 272);
        if ( !v24 )
        {
          ++*(_QWORD *)(a1 + 248);
          v43 = 0;
          goto LABEL_54;
        }
        v25 = v42;
        v26 = 1;
        v27 = 0;
        v28 = *(_QWORD *)(a1 + 256);
        v29 = (v24 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
        v30 = (_QWORD *)(v28 + 16LL * v29);
        v31 = *v30;
        if ( v42 != *v30 )
        {
          while ( v31 != -4096 )
          {
            if ( !v27 && v31 == -8192 )
              v27 = v30;
            v29 = (v24 - 1) & (v26 + v29);
            v30 = (_QWORD *)(v28 + 16LL * v29);
            v31 = *v30;
            if ( v42 == *v30 )
              goto LABEL_23;
            ++v26;
          }
          if ( !v27 )
            v27 = v30;
          v39 = *(_DWORD *)(a1 + 264);
          ++*(_QWORD *)(a1 + 248);
          v40 = v39 + 1;
          v43 = v27;
          if ( 4 * (v39 + 1) < 3 * v24 )
          {
            if ( v24 - *(_DWORD *)(a1 + 268) - v40 > v24 >> 3 )
            {
LABEL_50:
              *(_DWORD *)(a1 + 264) = v40;
              if ( *v27 != -4096 )
                --*(_DWORD *)(a1 + 268);
              *v27 = v25;
              *((_DWORD *)v27 + 2) = 0;
              v41 = (int *)(v27 + 1);
              goto LABEL_24;
            }
LABEL_55:
            sub_B23080(a1 + 248, v24);
            sub_B1C700(a1 + 248, &v42, &v43);
            v25 = v42;
            v27 = v43;
            v40 = *(_DWORD *)(a1 + 264) + 1;
            goto LABEL_50;
          }
LABEL_54:
          v24 *= 2;
          goto LABEL_55;
        }
LABEL_23:
        v41 = (int *)(v30 + 1);
LABEL_24:
        v32 = sub_22077B0(0xD0u);
        v33 = *v41;
        *(_QWORD *)(v32 + 16) = v42;
        *(_DWORD *)(v32 + 24) = v33;
        *(_QWORD *)(v32 + 32) = v44[0];
        *(_QWORD *)(v32 + 40) = v44[1];
        *(_DWORD *)(v32 + 48) = v45;
        sub_C8CF70(v32 + 56, (void *)(v32 + 88), 8, (__int64)v49, (__int64)v46);
        *(_QWORD *)(v32 + 152) = v32 + 168;
        *(_QWORD *)(v32 + 160) = 0x400000000LL;
        *(_WORD *)(v32 + 200) = 0;
        sub_2208C80((_QWORD *)v32, a1 + 128);
        ++*(_QWORD *)(a1 + 144);
        if ( !v48 )
          _libc_free(v47);
        v36 = *(_QWORD *)(a1 + 136) + 16LL;
        v37 = *(unsigned int *)(a2 + 56);
        if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 60) )
        {
          sub_C8D5F0(a2 + 48, (const void *)(a2 + 64), v37 + 1, 8u, v34, v35);
          v37 = *(unsigned int *)(a2 + 56);
        }
        *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8 * v37) = v36;
        ++*(_DWORD *)(a2 + 56);
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
      }
      else
      {
LABEL_17:
        sub_C8CC70((__int64)&v50, v42, (__int64)v6, v7, v8, v9);
        if ( v20 )
        {
          v21 = *(unsigned __int8 **)a2;
          v22 = **(_BYTE **)a2;
          if ( v22 > 0x1Cu )
            goto LABEL_19;
          goto LABEL_32;
        }
LABEL_8:
        v5 = *(_QWORD *)(v5 + 8);
        if ( !v5 )
          break;
      }
    }
  }
  v11 = *(char **)(a2 + 48);
  v12 = 8LL * *(unsigned int *)(a2 + 56);
  v13 = &v11[v12];
  if ( v11 == &v11[v12] )
  {
LABEL_15:
    if ( v54 )
      return;
LABEL_37:
    _libc_free((unsigned __int64)v51);
    return;
  }
  _BitScanReverse64(&v14, v12 >> 3);
  sub_2D04260(*(char **)(a2 + 48), (__int64 *)&v11[v12], 2LL * (int)(63 - (v14 ^ 0x3F)));
  if ( (unsigned __int64)v12 > 0x80 )
  {
    v15 = v11 + 128;
    sub_2D04090(v11, v11 + 128);
    if ( v13 != v11 + 128 )
    {
      do
      {
        while ( 1 )
        {
          v16 = *((_QWORD *)v15 - 1);
          v17 = *(_QWORD *)v15;
          v18 = v15 - 8;
          if ( *(_DWORD *)(*(_QWORD *)v15 + 8LL) < *(_DWORD *)(v16 + 8) )
            break;
          v38 = v15;
          v15 += 8;
          *(_QWORD *)v38 = v17;
          if ( v13 == v15 )
            goto LABEL_15;
        }
        do
        {
          *((_QWORD *)v18 + 1) = v16;
          v19 = v18;
          v16 = *((_QWORD *)v18 - 1);
          v18 -= 8;
        }
        while ( *(_DWORD *)(v17 + 8) < *(_DWORD *)(v16 + 8) );
        v15 += 8;
        *(_QWORD *)v19 = v17;
      }
      while ( v13 != v15 );
    }
    goto LABEL_15;
  }
  sub_2D04090(v11, &v11[v12]);
  if ( !v54 )
    goto LABEL_37;
}
