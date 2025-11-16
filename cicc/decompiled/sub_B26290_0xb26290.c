// Function: sub_B26290
// Address: 0xb26290
//
__int64 __fastcall sub_B26290(__int64 a1, unsigned __int64 *a2, __int64 a3, unsigned __int8 a4)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  __int64 *v7; // r15
  __int64 v8; // r8
  int v9; // esi
  unsigned int v10; // eax
  _QWORD *v11; // rdi
  __int64 v12; // r9
  _QWORD *v13; // rdi
  __int64 v14; // r12
  __int64 v15; // r14
  char v16; // dl
  bool v17; // r13
  int v18; // edx
  __int64 v19; // r8
  int v20; // esi
  unsigned int v21; // eax
  _QWORD *v22; // rdi
  __int64 v23; // r9
  _QWORD *v24; // rdx
  unsigned __int64 v25; // r12
  __int64 v26; // r13
  char v27; // dl
  int v28; // edx
  unsigned int v29; // esi
  unsigned int v30; // esi
  unsigned int v31; // eax
  int v32; // edi
  unsigned int v33; // r8d
  __int64 v34; // rax
  unsigned int v35; // eax
  int v36; // edi
  unsigned int v37; // r8d
  unsigned int v38; // edx
  __int64 v39; // rdi
  __int64 v40; // rax
  int v42; // ecx
  _QWORD *v43; // r10
  int v44; // r11d
  _QWORD *v45; // r10
  __int64 v46; // [rsp+0h] [rbp-70h]
  __int64 *v48; // [rsp+20h] [rbp-50h]
  __int64 v49; // [rsp+28h] [rbp-48h]
  __int64 v50; // [rsp+30h] [rbp-40h] BYREF
  _QWORD v51[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = (_QWORD *)(a1 + 16);
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 1;
  v46 = a1 + 304;
  do
  {
    if ( v5 )
      *v5 = -4096;
    v5 += 9;
  }
  while ( v5 != (_QWORD *)(a1 + 304) );
  v6 = (_QWORD *)(a1 + 320);
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 1;
  do
  {
    if ( v6 )
      *v6 = -4096;
    v6 += 9;
  }
  while ( v6 != (_QWORD *)(a1 + 608) );
  *(_QWORD *)(a1 + 616) = a1 + 632;
  *(_QWORD *)(a1 + 624) = 0x400000000LL;
  sub_B25B40(a2, a3, a1 + 616, 0, 0);
  v7 = *(__int64 **)(a1 + 616);
  v48 = &v7[2 * *(unsigned int *)(a1 + 624)];
  while ( v48 != v7 )
  {
    v14 = v7[1];
    v15 = *v7;
    v16 = *(_BYTE *)(a1 + 8);
    v50 = *v7;
    v17 = !((v14 >> 2) & 1) == (bool)(a4 ^ 1);
    v18 = v16 & 1;
    if ( v18 )
    {
      v19 = a1 + 16;
      v20 = 3;
    }
    else
    {
      v30 = *(_DWORD *)(a1 + 24);
      v19 = *(_QWORD *)(a1 + 16);
      if ( !v30 )
      {
        v35 = *(_DWORD *)(a1 + 8);
        ++*(_QWORD *)a1;
        v51[0] = 0;
        v36 = (v35 >> 1) + 1;
LABEL_32:
        v37 = 3 * v30;
        goto LABEL_33;
      }
      v20 = v30 - 1;
    }
    v21 = v20 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v22 = (_QWORD *)(v19 + 72LL * v21);
    v23 = *v22;
    if ( v15 == *v22 )
    {
LABEL_18:
      v24 = v22 + 1;
      goto LABEL_19;
    }
    v44 = 1;
    v45 = 0;
    while ( v23 != -4096 )
    {
      if ( v23 == -8192 && !v45 )
        v45 = v22;
      v21 = v20 & (v44 + v21);
      v22 = (_QWORD *)(v19 + 72LL * v21);
      v23 = *v22;
      if ( v15 == *v22 )
        goto LABEL_18;
      ++v44;
    }
    v35 = *(_DWORD *)(a1 + 8);
    v37 = 12;
    v30 = 4;
    if ( !v45 )
      v45 = v22;
    ++*(_QWORD *)a1;
    v51[0] = v45;
    v36 = (v35 >> 1) + 1;
    if ( !(_BYTE)v18 )
    {
      v30 = *(_DWORD *)(a1 + 24);
      goto LABEL_32;
    }
LABEL_33:
    if ( 4 * v36 >= v37 )
    {
      v30 *= 2;
LABEL_40:
      sub_B21EE0(a1, v30);
      sub_B1BD90(a1, &v50, v51);
      v39 = v50;
      v35 = *(_DWORD *)(a1 + 8);
      goto LABEL_35;
    }
    v38 = v30 - *(_DWORD *)(a1 + 12) - v36;
    v39 = v15;
    if ( v38 <= v30 >> 3 )
      goto LABEL_40;
LABEL_35:
    *(_DWORD *)(a1 + 8) = (2 * (v35 >> 1) + 2) | v35 & 1;
    v40 = v51[0];
    if ( *(_QWORD *)v51[0] != -4096 )
      --*(_DWORD *)(a1 + 12);
    *(_QWORD *)v40 = v39;
    v24 = (_QWORD *)(v40 + 8);
    *(_QWORD *)(v40 + 8) = v40 + 24;
    *(_QWORD *)(v40 + 16) = 0x200000000LL;
    *(_QWORD *)(v40 + 40) = v40 + 56;
    *(_QWORD *)(v40 + 48) = 0x200000000LL;
    *(_OWORD *)(v40 + 24) = 0;
    *(_OWORD *)(v40 + 56) = 0;
LABEL_19:
    v25 = v14 & 0xFFFFFFFFFFFFFFF8LL;
    v26 = 4LL * v17;
    v49 = v25;
    sub_B1A4E0((__int64)&v24[v26], v25);
    v27 = *(_BYTE *)(a1 + 312);
    v50 = v25;
    v28 = v27 & 1;
    if ( v28 )
    {
      v8 = a1 + 320;
      v9 = 3;
    }
    else
    {
      v29 = *(_DWORD *)(a1 + 328);
      v8 = *(_QWORD *)(a1 + 320);
      if ( !v29 )
      {
        v31 = *(_DWORD *)(a1 + 312);
        ++*(_QWORD *)(a1 + 304);
        v51[0] = 0;
        v32 = (v31 >> 1) + 1;
        goto LABEL_25;
      }
      v9 = v29 - 1;
    }
    v10 = v9 & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
    v11 = (_QWORD *)(v8 + 72LL * v10);
    v12 = *v11;
    if ( v25 != *v11 )
    {
      v42 = 1;
      v43 = 0;
      while ( v12 != -4096 )
      {
        if ( v12 == -8192 && !v43 )
          v43 = v11;
        v10 = v9 & (v42 + v10);
        v11 = (_QWORD *)(v8 + 72LL * v10);
        v12 = *v11;
        if ( v25 == *v11 )
          goto LABEL_13;
        ++v42;
      }
      v31 = *(_DWORD *)(a1 + 312);
      v33 = 12;
      v29 = 4;
      if ( !v43 )
        v43 = v11;
      ++*(_QWORD *)(a1 + 304);
      v51[0] = v43;
      v32 = (v31 >> 1) + 1;
      if ( !(_BYTE)v28 )
      {
        v29 = *(_DWORD *)(a1 + 328);
LABEL_25:
        v33 = 3 * v29;
      }
      if ( 4 * v32 >= v33 )
      {
        v29 *= 2;
      }
      else if ( v29 - *(_DWORD *)(a1 + 316) - v32 > v29 >> 3 )
      {
LABEL_28:
        *(_DWORD *)(a1 + 312) = (2 * (v31 >> 1) + 2) | v31 & 1;
        v34 = v51[0];
        if ( *(_QWORD *)v51[0] != -4096 )
          --*(_DWORD *)(a1 + 316);
        v13 = (_QWORD *)(v34 + 8);
        *(_QWORD *)(v34 + 8) = v34 + 24;
        *(_QWORD *)v34 = v49;
        *(_QWORD *)(v34 + 16) = 0x200000000LL;
        *(_QWORD *)(v34 + 40) = v34 + 56;
        *(_QWORD *)(v34 + 48) = 0x200000000LL;
        *(_OWORD *)(v34 + 24) = 0;
        *(_OWORD *)(v34 + 56) = 0;
        goto LABEL_14;
      }
      sub_B21EE0(v46, v29);
      sub_B1BD90(v46, &v50, v51);
      v49 = v50;
      v31 = *(_DWORD *)(a1 + 312);
      goto LABEL_28;
    }
LABEL_13:
    v13 = v11 + 1;
LABEL_14:
    v7 += 2;
    sub_B1A4E0((__int64)&v13[v26], v15);
  }
  *(_BYTE *)(a1 + 608) = a4;
  return a4;
}
