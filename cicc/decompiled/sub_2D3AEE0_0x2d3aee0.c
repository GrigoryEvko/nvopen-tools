// Function: sub_2D3AEE0
// Address: 0x2d3aee0
//
void __fastcall sub_2D3AEE0(__int64 a1, __int64 a2, __int64 a3, int a4, int a5, int a6, int a7, __int64 *a8)
{
  __int64 v11; // rsi
  unsigned int v12; // esi
  __int64 v13; // rcx
  int v14; // r9d
  unsigned int v15; // edx
  __int64 *v16; // r13
  __int64 *v17; // rax
  __int64 v18; // r8
  unsigned int v19; // esi
  __int64 v20; // rbx
  __int64 v21; // r8
  __int64 v22; // rdi
  __int64 v23; // r9
  int v24; // r10d
  unsigned int v25; // edx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rbx
  __int64 v30; // rdx
  unsigned __int64 v31; // rcx
  unsigned __int64 v32; // rsi
  int v33; // eax
  __int64 v34; // rsi
  _QWORD *v35; // rcx
  _QWORD *v36; // rdi
  __int64 v37; // rax
  int v38; // eax
  int v39; // ecx
  __int64 **v40; // rdx
  __int64 *v41; // r13
  __int64 v42; // rcx
  unsigned __int64 v43; // rdi
  unsigned __int64 v44; // rsi
  int v45; // eax
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 **v48; // rdi
  _BYTE *v49; // r14
  _BYTE *v50; // r12
  __int64 v51; // rsi
  _BYTE *v52; // r15
  _BYTE *v53; // r12
  __int64 v54; // rsi
  int v55; // ecx
  int v56; // edi
  __int64 **v57; // rdx
  unsigned __int64 v58; // r13
  __int64 v59; // rdi
  unsigned __int64 v60; // r12
  __int64 v61; // rdi
  __int64 **v62; // [rsp+0h] [rbp-110h]
  __int64 **v63; // [rsp+8h] [rbp-108h]
  __int64 v64; // [rsp+18h] [rbp-F8h] BYREF
  __int64 v65; // [rsp+20h] [rbp-F0h] BYREF
  int v66; // [rsp+28h] [rbp-E8h]
  _DWORD v67[4]; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v68[2]; // [rsp+40h] [rbp-D0h] BYREF
  _BYTE *v69; // [rsp+50h] [rbp-C0h]
  __int64 v70; // [rsp+58h] [rbp-B8h]
  _BYTE v71[48]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 *v72; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v73; // [rsp+98h] [rbp-78h]
  __int64 v74; // [rsp+A0h] [rbp-70h]
  _BYTE v75[104]; // [rsp+A8h] [rbp-68h] BYREF

  if ( !a7 )
    return;
  v67[1] = a7;
  v67[0] = a4;
  v11 = *a8;
  v67[2] = a5;
  v67[3] = a6 - a5;
  v68[0] = v11;
  if ( v11 )
    sub_B96E90((__int64)v68, v11, 1);
  v12 = *(_DWORD *)(a1 + 368);
  v64 = a2;
  if ( !v12 )
  {
    ++*(_QWORD *)(a1 + 344);
    v72 = 0;
    goto LABEL_72;
  }
  v13 = *(_QWORD *)(a1 + 352);
  v14 = 1;
  v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v16 = (__int64 *)(v13 + 56LL * v15);
  v17 = 0;
  v18 = *v16;
  if ( a2 != *v16 )
  {
    while ( v18 != -4096 )
    {
      if ( v18 == -8192 && !v17 )
        v17 = v16;
      v15 = (v12 - 1) & (v14 + v15);
      v16 = (__int64 *)(v13 + 56LL * v15);
      v18 = *v16;
      if ( a2 == *v16 )
        goto LABEL_7;
      ++v14;
    }
    v55 = *(_DWORD *)(a1 + 360);
    if ( !v17 )
      v17 = v16;
    ++*(_QWORD *)(a1 + 344);
    v56 = v55 + 1;
    v72 = v17;
    if ( 4 * (v55 + 1) < 3 * v12 )
    {
      v57 = &v72;
      if ( v12 - *(_DWORD *)(a1 + 364) - v56 > v12 >> 3 )
      {
LABEL_62:
        *(_DWORD *)(a1 + 360) = v56;
        if ( *v17 != -4096 )
          --*(_DWORD *)(a1 + 364);
        *v17 = a2;
        v20 = (__int64)(v17 + 1);
        *(_OWORD *)(v17 + 1) = 0;
        v17[5] = (__int64)(v17 + 7);
        v17[6] = 0;
        v65 = a3;
        v66 = 0;
        *(_OWORD *)(v17 + 3) = 0;
        goto LABEL_65;
      }
      sub_2D267E0(a1 + 344, v12);
LABEL_73:
      sub_2D228B0(a1 + 344, &v64, &v72);
      a2 = v64;
      v56 = *(_DWORD *)(a1 + 360) + 1;
      v17 = v72;
      goto LABEL_62;
    }
LABEL_72:
    sub_2D267E0(a1 + 344, 2 * v12);
    goto LABEL_73;
  }
LABEL_7:
  v19 = *((_DWORD *)v16 + 8);
  v65 = a3;
  v20 = (__int64)(v16 + 1);
  v66 = 0;
  if ( !v19 )
  {
    v57 = &v72;
LABEL_65:
    v72 = 0;
    v19 = 0;
    ++*(_QWORD *)v20;
    goto LABEL_66;
  }
  v21 = v19 - 1;
  v22 = v16[2];
  v23 = 0;
  v24 = 1;
  v25 = v21 & (37 * a3);
  v26 = v22 + 16LL * v25;
  v27 = *(_QWORD *)v26;
  if ( a3 != *(_QWORD *)v26 )
  {
    while ( v27 != -4096 )
    {
      if ( !v23 && v27 == -8192 )
        v23 = v26;
      v25 = v21 & (v24 + v25);
      v26 = v22 + 16LL * v25;
      v27 = *(_QWORD *)v26;
      if ( a3 == *(_QWORD *)v26 )
        goto LABEL_9;
      ++v24;
    }
    if ( !v23 )
      v23 = v26;
    v72 = (__int64 *)v23;
    v38 = *((_DWORD *)v16 + 6);
    ++v16[1];
    v39 = v38 + 1;
    if ( 4 * (v38 + 1) < 3 * v19 )
    {
      v40 = &v72;
      if ( v19 - *((_DWORD *)v16 + 7) - v39 <= v19 >> 3 )
      {
        sub_2D36220((__int64)(v16 + 1), v19);
        sub_2D2BD40((__int64)(v16 + 1), &v65, &v72);
        v40 = &v72;
        v39 = *((_DWORD *)v16 + 6) + 1;
      }
      goto LABEL_29;
    }
    v57 = &v72;
LABEL_66:
    v63 = v57;
    sub_2D36220(v20, 2 * v19);
    sub_2D2BD40(v20, &v65, v63);
    v40 = v63;
    v39 = *(_DWORD *)(v20 + 16) + 1;
LABEL_29:
    *(_DWORD *)(v20 + 16) = v39;
    v41 = v72;
    if ( *v72 != -4096 )
      --*(_DWORD *)(v20 + 20);
    *v41 = v65;
    *((_DWORD *)v41 + 2) = v66;
    v72 = (__int64 *)a3;
    v69 = v71;
    v74 = 0x200000000LL;
    v73 = v75;
    v42 = *(unsigned int *)(v20 + 40);
    v43 = *(unsigned int *)(v20 + 44);
    v70 = 0x200000000LL;
    v44 = v42 + 1;
    v45 = v42;
    if ( v42 + 1 > v43 )
    {
      v60 = *(_QWORD *)(v20 + 32);
      v62 = v40;
      v61 = v20 + 32;
      if ( v60 > (unsigned __int64)v40 || (unsigned __int64)v40 >= v60 + 72 * v42 )
      {
        sub_2D26690(v61, v44, (__int64)v40, v42, v21, v23);
        v42 = *(unsigned int *)(v20 + 40);
        v46 = *(_QWORD *)(v20 + 32);
        v40 = v62;
        v45 = *(_DWORD *)(v20 + 40);
      }
      else
      {
        sub_2D26690(v61, v44, (__int64)v40, v42, v21, v23);
        v46 = *(_QWORD *)(v20 + 32);
        v42 = *(unsigned int *)(v20 + 40);
        v40 = (__int64 **)((char *)v62 + v46 - v60);
        v45 = *(_DWORD *)(v20 + 40);
      }
    }
    else
    {
      v46 = *(_QWORD *)(v20 + 32);
    }
    v47 = 9 * v42;
    v48 = (__int64 **)(v46 + 8 * v47);
    if ( v48 )
    {
      *v48 = *v40;
      v48[1] = (__int64 *)(v48 + 3);
      v48[2] = (__int64 *)0x200000000LL;
      if ( *((_DWORD *)v40 + 4) )
        sub_2D262B0((__int64)(v48 + 1), (__int64)(v40 + 1), (__int64)v40, v47, v21, v23);
      v45 = *(_DWORD *)(v20 + 40);
    }
    *(_DWORD *)(v20 + 40) = v45 + 1;
    v49 = v73;
    v50 = &v73[24 * (unsigned int)v74];
    if ( v73 != v50 )
    {
      do
      {
        v51 = *((_QWORD *)v50 - 1);
        v50 -= 24;
        if ( v51 )
          sub_B91220((__int64)(v50 + 16), v51);
      }
      while ( v49 != v50 );
      v50 = v73;
    }
    if ( v50 != v75 )
      _libc_free((unsigned __int64)v50);
    v52 = v69;
    v53 = &v69[24 * (unsigned int)v70];
    if ( v69 != v53 )
    {
      do
      {
        v54 = *((_QWORD *)v53 - 1);
        v53 -= 24;
        if ( v54 )
          sub_B91220((__int64)(v53 + 16), v54);
      }
      while ( v52 != v53 );
      v53 = v69;
    }
    if ( v53 != v71 )
      _libc_free((unsigned __int64)v53);
    v28 = (unsigned int)(*(_DWORD *)(v20 + 40) - 1);
    *((_DWORD *)v41 + 2) = v28;
    goto LABEL_10;
  }
LABEL_9:
  v28 = *(unsigned int *)(v26 + 8);
LABEL_10:
  v29 = *(_QWORD *)(v20 + 32) + 72 * v28;
  v30 = *(unsigned int *)(v29 + 16);
  v31 = *(unsigned int *)(v29 + 20);
  v32 = v30 + 1;
  v33 = *(_DWORD *)(v29 + 16);
  if ( v30 + 1 > v31 )
  {
    v58 = *(_QWORD *)(v29 + 8);
    v59 = v29 + 8;
    if ( v58 > (unsigned __int64)v67 || (unsigned __int64)v67 >= v58 + 24 * v30 )
    {
      sub_2D24250(v59, v32, v30, v31, v21, v23);
      v30 = *(unsigned int *)(v29 + 16);
      v34 = *(_QWORD *)(v29 + 8);
      v35 = v67;
      v33 = *(_DWORD *)(v29 + 16);
    }
    else
    {
      sub_2D24250(v59, v32, v30, v31, v21, v23);
      v34 = *(_QWORD *)(v29 + 8);
      v30 = *(unsigned int *)(v29 + 16);
      v35 = (_QWORD *)((char *)v67 + v34 - v58);
      v33 = *(_DWORD *)(v29 + 16);
    }
  }
  else
  {
    v34 = *(_QWORD *)(v29 + 8);
    v35 = v67;
  }
  v36 = (_QWORD *)(v34 + 24 * v30);
  if ( v36 )
  {
    *v36 = *v35;
    v36[1] = v35[1];
    v37 = v35[2];
    v36[2] = v37;
    if ( v37 )
      sub_2D23AB0(v36 + 2);
    v33 = *(_DWORD *)(v29 + 16);
  }
  *(_DWORD *)(v29 + 16) = v33 + 1;
  if ( v68[0] )
    sub_B91220((__int64)v68, v68[0]);
}
