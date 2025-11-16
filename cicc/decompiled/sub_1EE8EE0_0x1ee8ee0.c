// Function: sub_1EE8EE0
// Address: 0x1ee8ee0
//
void __fastcall sub_1EE8EE0(__int64 a1, unsigned __int64 a2)
{
  __int64 v3; // rcx
  __int64 v4; // rdx
  char v5; // r8
  int v6; // r12d
  int v7; // r14d
  unsigned __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 (*v12)(void); // rax
  __int64 v13; // rax
  char v14; // dl
  __int64 v15; // rsi
  unsigned __int64 j; // rdx
  __int64 v17; // r8
  __int64 v18; // rsi
  unsigned int v19; // edi
  __int64 *v20; // rcx
  __int64 v21; // r10
  unsigned int v22; // edx
  __int64 v23; // rsi
  unsigned int v24; // eax
  __int64 v25; // rdi
  _DWORD *v26; // rdx
  int v27; // edx
  int v28; // ecx
  int *v29; // r12
  int *v30; // rbx
  int v31; // r8d
  unsigned int v32; // esi
  __int64 v33; // rcx
  unsigned int v34; // eax
  __int64 v35; // rdi
  _DWORD *v36; // rdx
  int v37; // edx
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // rax
  unsigned int v41; // ecx
  unsigned __int64 v42; // rdx
  __int64 i; // rcx
  unsigned int v44; // edi
  __int64 v45; // r8
  unsigned int v46; // ecx
  __int64 *v47; // rax
  __int64 v48; // r10
  int v49; // eax
  int v50; // r11d
  _QWORD *v51; // [rsp+0h] [rbp-170h]
  int v52; // [rsp+0h] [rbp-170h]
  int *v53; // [rsp+20h] [rbp-150h]
  signed __int64 v54; // [rsp+28h] [rbp-148h]
  __int64 v55; // [rsp+30h] [rbp-140h]
  int *v56; // [rsp+48h] [rbp-128h]
  int *v57; // [rsp+50h] [rbp-120h] BYREF
  __int64 v58; // [rsp+58h] [rbp-118h]
  _BYTE v59[64]; // [rsp+60h] [rbp-110h] BYREF
  int *v60; // [rsp+A0h] [rbp-D0h]
  __int64 v61; // [rsp+A8h] [rbp-C8h]
  _BYTE v62[64]; // [rsp+B0h] [rbp-C0h] BYREF
  int *v63; // [rsp+F0h] [rbp-80h]
  __int64 v64; // [rsp+F8h] [rbp-78h]
  _BYTE v65[112]; // [rsp+100h] [rbp-70h] BYREF

  v54 = 0;
  if ( *(_BYTE *)(a1 + 56) )
  {
    v42 = a2;
    for ( i = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 272LL);
          (*(_BYTE *)(v42 + 46) & 4) != 0;
          v42 = *(_QWORD *)v42 & 0xFFFFFFFFFFFFFFF8LL )
    {
      ;
    }
    v44 = *(_DWORD *)(i + 384);
    v45 = *(_QWORD *)(i + 368);
    if ( v44 )
    {
      v46 = (v44 - 1) & (((unsigned int)v42 >> 9) ^ ((unsigned int)v42 >> 4));
      v47 = (__int64 *)(v45 + 16LL * v46);
      v48 = *v47;
      if ( v42 == *v47 )
      {
LABEL_65:
        v54 = v47[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
        goto LABEL_2;
      }
      v49 = 1;
      while ( v48 != -8 )
      {
        v50 = v49 + 1;
        v46 = (v44 - 1) & (v49 + v46);
        v47 = (__int64 *)(v45 + 16LL * v46);
        v48 = *v47;
        if ( *v47 == v42 )
          goto LABEL_65;
        v49 = v50;
      }
    }
    v47 = (__int64 *)(v45 + 16LL * v44);
    goto LABEL_65;
  }
LABEL_2:
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 8);
  v5 = *(_BYTE *)(a1 + 58);
  v57 = (int *)v59;
  v60 = (int *)v62;
  v58 = 0x800000000LL;
  v61 = 0x800000000LL;
  v63 = (int *)v65;
  v64 = 0x800000000LL;
  sub_1EE65F0((__int64)&v57, a2, v4, v3, v5, 0);
  if ( *(_BYTE *)(a1 + 58) )
    sub_1EE6D60((__int64)&v57, *(_QWORD *)(a1 + 32), *(_QWORD *)(a1 + 24), v54, 0);
  if ( *(_BYTE *)(a1 + 56) )
  {
    v53 = &v57[2 * (unsigned int)v58];
    if ( v57 != v53 )
    {
      v56 = v57;
      while ( 1 )
      {
        v6 = *v56;
        v7 = sub_1EE80B0(a1, *v56, v54);
        if ( v7 )
          break;
LABEL_36:
        v56 += 2;
        if ( v53 == v56 )
          goto LABEL_37;
      }
      v8 = sub_1EE6230((_QWORD *)a1);
      v9 = *(_QWORD **)(a1 + 24);
      v10 = *(_QWORD *)(a1 + 32);
      v55 = 0;
      v11 = v8;
      v12 = *(__int64 (**)(void))(**(_QWORD **)(*v9 + 16LL) + 112LL);
      if ( v12 != sub_1D00B10 )
      {
        v51 = *(_QWORD **)(a1 + 24);
        v40 = v12();
        v9 = v51;
        v55 = v40;
      }
      if ( v6 < 0 )
      {
        v13 = *(_QWORD *)(v9[3] + 16LL * (v6 & 0x7FFFFFFF) + 8);
        if ( v13 )
          goto LABEL_14;
LABEL_59:
        v41 = *(_DWORD *)(a1 + 192) + (v6 & 0x7FFFFFFF);
        goto LABEL_29;
      }
      v13 = *(_QWORD *)(v9[34] + 8LL * (unsigned int)v6);
      if ( !v13 )
      {
        v41 = v6;
LABEL_29:
        v23 = *(unsigned int *)(a1 + 104);
        v24 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v41);
        if ( v24 >= (unsigned int)v23 )
          goto LABEL_60;
        v25 = *(_QWORD *)(a1 + 96);
        while ( 1 )
        {
          v26 = (_DWORD *)(v25 + 8LL * v24);
          if ( v41 == *v26 )
            break;
          v24 += 256;
          if ( (unsigned int)v23 <= v24 )
            goto LABEL_60;
        }
        if ( v26 == (_DWORD *)(v25 + 8 * v23) )
        {
LABEL_60:
          v28 = 0;
          v27 = 0;
        }
        else
        {
          v27 = v26[1];
          v28 = v27 & ~v7;
        }
        sub_1EE5E20(a1, v6, v27, v28);
        goto LABEL_36;
      }
LABEL_14:
      while ( 1 )
      {
        if ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
        {
          v14 = *(_BYTE *)(v13 + 4);
          if ( (v14 & 8) == 0 )
            break;
        }
        v13 = *(_QWORD *)(v13 + 32);
        if ( !v13 )
          goto LABEL_28;
      }
LABEL_16:
      if ( (v14 & 1) != 0 )
      {
LABEL_24:
        while ( 1 )
        {
          v13 = *(_QWORD *)(v13 + 32);
          if ( !v13 )
            break;
          while ( (*(_BYTE *)(v13 + 3) & 0x10) == 0 )
          {
            v14 = *(_BYTE *)(v13 + 4);
            if ( (v14 & 8) == 0 )
              goto LABEL_16;
            v13 = *(_QWORD *)(v13 + 32);
            if ( !v13 )
              goto LABEL_28;
          }
        }
LABEL_28:
        v41 = v6;
        if ( v6 < 0 )
          goto LABEL_59;
        goto LABEL_29;
      }
      v15 = *(_QWORD *)(v10 + 272);
      for ( j = *(_QWORD *)(v13 + 16); (*(_BYTE *)(j + 46) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
        ;
      v17 = *(_QWORD *)(v15 + 368);
      v18 = *(unsigned int *)(v15 + 384);
      if ( (_DWORD)v18 )
      {
        v19 = (v18 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
        v20 = (__int64 *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( *v20 == j )
        {
LABEL_21:
          v22 = *(_DWORD *)((v20[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | 2;
          if ( v22 >= ((unsigned int)(v11 >> 1) & 3 | *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24))
            && v22 < (*(_DWORD *)((v54 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v54 >> 1) & 2) )
          {
            v7 &= ~*(_DWORD *)(*(_QWORD *)(v55 + 248) + 4LL * ((*(_DWORD *)v13 >> 8) & 0xFFF));
            if ( !v7 )
              goto LABEL_36;
          }
          goto LABEL_24;
        }
        v39 = 1;
        while ( v21 != -8 )
        {
          v19 = (v18 - 1) & (v39 + v19);
          v52 = v39 + 1;
          v20 = (__int64 *)(v17 + 16LL * v19);
          v21 = *v20;
          if ( *v20 == j )
            goto LABEL_21;
          v39 = v52;
        }
      }
      v20 = (__int64 *)(v17 + 16 * v18);
      goto LABEL_21;
    }
  }
LABEL_37:
  v29 = v60;
  v30 = &v60[2 * (unsigned int)v61];
  if ( v30 != v60 )
  {
    do
    {
      v31 = *v29;
      v32 = *v29;
      if ( *v29 < 0 )
        v32 = *(_DWORD *)(a1 + 192) + (v32 & 0x7FFFFFFF);
      v33 = *(unsigned int *)(a1 + 104);
      v34 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 176) + v32);
      if ( v34 >= (unsigned int)v33 )
        goto LABEL_54;
      v35 = *(_QWORD *)(a1 + 96);
      while ( 1 )
      {
        v36 = (_DWORD *)(v35 + 8LL * v34);
        if ( v32 == *v36 )
          break;
        v34 += 256;
        if ( (unsigned int)v33 <= v34 )
          goto LABEL_54;
      }
      if ( v36 == (_DWORD *)(v35 + 8 * v33) )
LABEL_54:
        v37 = 0;
      else
        v37 = v36[1];
      v38 = v29[1];
      v29 += 2;
      sub_1EE5D10(a1, v31, v37, v37 | v38);
    }
    while ( v30 != v29 );
  }
  sub_1EE7580(a1, v63, (unsigned int)v64);
  if ( v63 != (int *)v65 )
    _libc_free((unsigned __int64)v63);
  if ( v60 != (int *)v62 )
    _libc_free((unsigned __int64)v60);
  if ( v57 != (int *)v59 )
    _libc_free((unsigned __int64)v57);
}
