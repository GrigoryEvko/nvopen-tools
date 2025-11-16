// Function: sub_35A8C50
// Address: 0x35a8c50
//
__int64 __fastcall sub_35A8C50(__int64 *a1)
{
  __int64 v2; // rsi
  void (__fastcall *v3)(_BYTE **, __int64, __int64); // rcx
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // r12
  int v8; // r15d
  unsigned int v9; // r13d
  __int64 v10; // r15
  _QWORD *v11; // rax
  unsigned __int64 v12; // rdi
  _QWORD *v13; // rdx
  __int64 v14; // rbx
  _QWORD *v15; // rax
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  __int64 v18; // rcx
  _QWORD *v19; // rax
  __int64 *v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // r15
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // r8
  __int64 *v30; // r15
  __int64 *v31; // r10
  __int64 v32; // rbx
  __int64 v33; // rcx
  __int64 *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // r8
  __int64 v38; // r9
  unsigned __int64 v39; // r15
  __int64 *v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 v42; // rcx
  __int64 v43; // rcx
  __int64 v44; // rcx
  _QWORD *v45; // r8
  __int64 v46; // rsi
  __int64 j; // r12
  __int64 v48; // rsi
  __int64 v49; // rdi
  __int64 k; // rbx
  __int64 v51; // rsi
  __int64 v52; // rdi
  _QWORD *v54; // rax
  __int64 v55; // [rsp+10h] [rbp-100h]
  __int64 v56; // [rsp+28h] [rbp-E8h]
  __int64 *v57; // [rsp+30h] [rbp-E0h]
  unsigned int v58; // [rsp+48h] [rbp-C8h]
  __int64 v59; // [rsp+48h] [rbp-C8h]
  __int64 v60; // [rsp+48h] [rbp-C8h]
  __int64 v61; // [rsp+48h] [rbp-C8h]
  unsigned __int64 i; // [rsp+58h] [rbp-B8h] BYREF
  __int64 v63; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v64; // [rsp+68h] [rbp-A8h]
  __int64 v65; // [rsp+70h] [rbp-A0h]
  unsigned int v66; // [rsp+78h] [rbp-98h]
  _BYTE *v67; // [rsp+80h] [rbp-90h] BYREF
  __int64 v68; // [rsp+88h] [rbp-88h]
  _BYTE v69[32]; // [rsp+90h] [rbp-80h] BYREF
  _BYTE *v70; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v71; // [rsp+B8h] [rbp-58h]
  _BYTE v72[80]; // [rsp+C0h] [rbp-50h] BYREF

  v2 = a1[4];
  v3 = *(void (__fastcall **)(_BYTE **, __int64, __int64))(*(_QWORD *)v2 + 376LL);
  v4 = 0;
  if ( (char *)v3 != (char *)sub_2FDC520 )
  {
    v3(&v70, v2, a1[6]);
    v4 = (__int64)v70;
  }
  v5 = a1[9];
  v70 = 0;
  a1[9] = v4;
  if ( v5 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v5 + 8LL))(v5);
    if ( v70 )
      (*(void (__fastcall **)(_BYTE *))(*(_QWORD *)v70 + 8LL))(v70);
  }
  v6 = a1[6];
  LOBYTE(v71) = 0;
  v7 = sub_2E7AAE0(a1[1], *(_QWORD *)(v6 + 16), (__int64)v70, 0);
  v8 = *(_DWORD *)(*a1 + 96);
  v9 = v8 - 1;
  v10 = (unsigned int)(2 * v8);
  v11 = (_QWORD *)sub_2207820(32 * v10 + 8);
  v12 = 32 * v10 + 8;
  v13 = v11;
  if ( !v11 )
  {
    v14 = 0;
    goto LABEL_39;
  }
  *v11 = v10;
  v14 = (__int64)(v11 + 1);
  if ( !v10 )
  {
LABEL_39:
    v54 = (_QWORD *)sub_2207820(v12);
    v55 = (__int64)v54;
    if ( v54 )
    {
      *v54 = v10;
      v18 = v10 - 1;
      v55 = (__int64)(v54 + 1);
      if ( v10 )
        goto LABEL_12;
    }
    goto LABEL_14;
  }
  v15 = v11 + 1;
  v16 = (_QWORD *)((char *)v13 + v12);
  do
  {
    *v15 = 0;
    v15 += 4;
    *((_DWORD *)v15 - 2) = 0;
    *(v15 - 3) = 0;
    *((_DWORD *)v15 - 4) = 0;
    *((_DWORD *)v15 - 3) = 0;
  }
  while ( v16 != v15 );
  v17 = (_QWORD *)sub_2207820(v12);
  v18 = v10 - 1;
  v55 = (__int64)v17;
  if ( v17 )
  {
    *v17 = v10;
    v55 = (__int64)(v17 + 1);
LABEL_12:
    v19 = (_QWORD *)v55;
    do
    {
      *v19 = 0;
      v19 += 4;
      *((_DWORD *)v19 - 2) = 0;
      *(v19 - 3) = 0;
      *((_DWORD *)v19 - 4) = 0;
      *((_DWORD *)v19 - 3) = 0;
    }
    while ( (_QWORD *)(v55 + 32 * v18 + 32) != v19 );
  }
LABEL_14:
  v67 = v69;
  v68 = 0x400000000LL;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  sub_35A7A10(a1, v9, v7, v14, (__int64)&v67);
  v20 = (__int64 *)a1[6];
  sub_2E33BD0(a1[1] + 320, v7);
  v21 = *v20;
  v22 = *(_QWORD *)v7;
  *(_QWORD *)(v7 + 8) = v20;
  v21 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v7 = v21 | v22 & 7;
  *(_QWORD *)(v21 + 8) = v7;
  *v20 = v7 | *v20 & 7;
  v23 = a1[5];
  sub_2E34D50(*(_QWORD *)(v23 + 32), (__int64 *)v7, v21, v24, v25, v26);
  v28 = *(unsigned int *)(v23 + 352);
  v29 = *(unsigned int *)(v23 + 192);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(v23 + 356) )
  {
    v61 = *(unsigned int *)(v23 + 192);
    sub_C8D5F0(v23 + 344, (const void *)(v23 + 360), v28 + 1, 8u, v29, v27);
    v28 = *(unsigned int *)(v23 + 352);
    v29 = v61;
  }
  *(_QWORD *)(*(_QWORD *)(v23 + 344) + 8 * v28) = v29;
  ++*(_DWORD *)(v23 + 352);
  v30 = *(__int64 **)(*a1 + 8);
  v31 = *(__int64 **)(*a1 + 16);
  if ( v30 != v31 )
  {
    v56 = v14;
    do
    {
      v32 = *v30;
      if ( *(_WORD *)(*v30 + 68) != 68 && *(_WORD *)(*v30 + 68) )
      {
        v57 = v31;
        v58 = sub_3598DB0(*a1, *v30);
        v70 = sub_3599350((__int64)a1, v32, v9, v58);
        sub_359F080(a1, (__int64)v70, 0, v9, v58, v56);
        v59 = (__int64)v70;
        sub_2E31040((__int64 *)(v7 + 40), (__int64)v70);
        v33 = *(_QWORD *)(v7 + 48);
        *(_QWORD *)(v59 + 8) = v7 + 48;
        v33 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v59 = v33 | *(_QWORD *)v59 & 7LL;
        *(_QWORD *)(v33 + 8) = v59;
        *(_QWORD *)(v7 + 48) = *(_QWORD *)(v7 + 48) & 7LL | v59;
        v34 = sub_359C4A0((__int64)&v63, (__int64 *)&v70);
        v31 = v57;
        *v34 = v32;
      }
      ++v30;
    }
    while ( v31 != v30 );
    v14 = v56;
  }
  v35 = a1[6];
  v39 = sub_2E313E0(v35);
  v40 = (__int64 *)&i;
  for ( i = v39; v35 + 48 != i; v39 = i )
  {
    v70 = sub_2E7B2C0((_QWORD *)a1[1], v39);
    sub_359F080(a1, (__int64)v70, 0, v9, 0, v14);
    v60 = (__int64)v70;
    sub_2E31040((__int64 *)(v7 + 40), (__int64)v70);
    v41 = *(_QWORD *)v60;
    v42 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v60 + 8) = v7 + 48;
    *(_QWORD *)v60 = v42 | v41 & 7;
    *(_QWORD *)(v42 + 8) = v60;
    *(_QWORD *)(v7 + 48) = *(_QWORD *)(v7 + 48) & 7LL | v60;
    *sub_359C4A0((__int64)&v63, (__int64 *)&v70) = v39;
    sub_2FD79B0((__int64 *)&i);
  }
  a1[8] = v7;
  sub_2E340B0(v7, a1[6], v36, (__int64)v40, v37, v38);
  sub_2E33690(v7, a1[6], v7);
  sub_359D220(a1, v7, *(_QWORD *)&v67[8 * (unsigned int)v68 - 8], v7, v7, v14, (__int64)&v63, v9, v9, 0);
  sub_359E620(a1, v7, *(_QWORD *)&v67[8 * (unsigned int)v68 - 8], v7, v7, v14, v55, (__int64)&v63, v9, v9, 0);
  v43 = a1[6];
  v71 = 0x400000000LL;
  v70 = v72;
  sub_35A82F0(a1, v9, v7, v43, v14, v55, (__int64)&v70, &v67);
  sub_359B620(a1, v7, (__int64 *)&v70);
  sub_359B340(a1, v7, (__int64)&v70, v44, v45);
  sub_359F240(a1, a1[7], (__int64)&v67, (_QWORD *)v7, &v70, v14);
  if ( v14 )
  {
    v46 = 32LL * *(_QWORD *)(v14 - 8);
    for ( j = v14 + v46; v14 != j; j -= 32 )
    {
      v48 = *(unsigned int *)(j - 8);
      v49 = *(_QWORD *)(j - 24);
      sub_C7D6A0(v49, 8 * v48, 4);
    }
    j_j_j___libc_free_0_0(v14 - 8);
  }
  if ( v55 )
  {
    for ( k = v55 + 32LL * *(_QWORD *)(v55 - 8); v55 != k; k -= 32 )
    {
      v51 = *(unsigned int *)(k - 8);
      v52 = *(_QWORD *)(k - 24);
      sub_C7D6A0(v52, 8 * v51, 4);
    }
    j_j_j___libc_free_0_0(v55 - 8);
  }
  if ( v70 != v72 )
    _libc_free((unsigned __int64)v70);
  if ( v67 != v69 )
    _libc_free((unsigned __int64)v67);
  return sub_C7D6A0(v64, 16LL * v66, 8);
}
