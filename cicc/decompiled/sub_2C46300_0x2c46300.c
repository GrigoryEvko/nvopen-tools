// Function: sub_2C46300
// Address: 0x2c46300
//
void __fastcall sub_2C46300(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r13
  int v8; // eax
  __int64 v9; // rdx
  size_t v10; // rdx
  void *v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  __int64 v21; // rsi
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int64 v28; // r14
  int *v29; // r13
  __int64 v30; // rsi
  int v31; // ebx
  __int64 v32; // rax
  __int64 v33; // r13
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // r15
  unsigned int v37; // eax
  _QWORD **v38; // r10
  char v39; // si
  _QWORD **v40; // rcx
  int v41; // edx
  __int64 v42; // rax
  _QWORD *v43; // rbx
  _QWORD *v44; // r13
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  unsigned int v47; // ecx
  unsigned int v48; // eax
  int v49; // eax
  unsigned __int64 v50; // rax
  __int64 v51; // rax
  int v52; // r14d
  __int64 v53; // r15
  int v54; // eax
  __int64 v55; // r9
  unsigned int v56; // esi
  int v57; // eax
  int *v58; // rdx
  int v59; // eax
  __int64 v60; // r8
  __int64 v61; // rax
  int v62; // r13d
  int v63; // r11d
  __int64 v64; // r15
  int *v65; // rbx
  unsigned int v66; // esi
  int v67; // eax
  int *v68; // rdx
  int v69; // eax
  int v71; // [rsp+1Ch] [rbp-144h] BYREF
  int *v72; // [rsp+20h] [rbp-140h] BYREF
  __int64 v73; // [rsp+28h] [rbp-138h]
  _BYTE v74[64]; // [rsp+30h] [rbp-130h] BYREF
  int *v75; // [rsp+70h] [rbp-F0h] BYREF
  int v76; // [rsp+78h] [rbp-E8h]
  __int64 v77; // [rsp+80h] [rbp-E0h]
  __int64 v78; // [rsp+88h] [rbp-D8h]
  __int64 v79; // [rsp+90h] [rbp-D0h]
  __int64 v80; // [rsp+98h] [rbp-C8h]
  _QWORD *v81; // [rsp+A0h] [rbp-C0h]
  __int64 v82; // [rsp+A8h] [rbp-B8h]
  __int64 v83; // [rsp+B0h] [rbp-B0h]
  char *v84; // [rsp+B8h] [rbp-A8h]
  __int64 v85; // [rsp+C0h] [rbp-A0h]
  int v86; // [rsp+C8h] [rbp-98h]
  char v87; // [rsp+CCh] [rbp-94h]
  char v88; // [rsp+D0h] [rbp-90h] BYREF
  __int64 v89; // [rsp+110h] [rbp-50h]
  _QWORD *v90; // [rsp+118h] [rbp-48h]
  __int64 v91; // [rsp+120h] [rbp-40h]
  unsigned int v92; // [rsp+128h] [rbp-38h]

  v6 = a1 + 112;
  v8 = *(_DWORD *)(a1 + 128);
  ++*(_QWORD *)(a1 + 112);
  v71 = a2;
  if ( !v8 )
  {
    if ( *(_DWORD *)(a1 + 132) )
    {
      v9 = *(unsigned int *)(a1 + 136);
      if ( (unsigned int)v9 <= 0x40 )
      {
LABEL_4:
        v10 = 4 * v9;
        v11 = *(void **)(a1 + 120);
        if ( v10 )
          memset(v11, 255, v10);
        *(_QWORD *)(a1 + 128) = 0;
        *(_DWORD *)(a1 + 152) = 0;
        goto LABEL_7;
      }
      sub_C7D6A0(*(_QWORD *)(a1 + 120), 4 * v9, 4);
      *(_QWORD *)(a1 + 120) = 0;
      *(_QWORD *)(a1 + 128) = 0;
      *(_DWORD *)(a1 + 136) = 0;
      *(_DWORD *)(a1 + 152) = 0;
    }
    else
    {
      *(_DWORD *)(a1 + 152) = 0;
    }
LABEL_7:
    v12 = 0;
    if ( !*(_DWORD *)(a1 + 156) )
    {
      sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), 1u, 4u, a5, a6);
      v12 = 4LL * *(unsigned int *)(a1 + 152);
    }
    *(_DWORD *)(*(_QWORD *)(a1 + 144) + v12) = v71;
    v13 = (unsigned int)(*(_DWORD *)(a1 + 152) + 1);
    *(_DWORD *)(a1 + 152) = v13;
    if ( (unsigned int)v13 <= 2 )
      goto LABEL_10;
    v64 = *(_QWORD *)(a1 + 144) + 4 * v13;
    v65 = *(int **)(a1 + 144);
    while ( (unsigned __int8)sub_22B31A0(v6, v65, &v72) )
    {
LABEL_69:
      if ( (int *)v64 == ++v65 )
        goto LABEL_10;
    }
    v66 = *(_DWORD *)(a1 + 136);
    v67 = *(_DWORD *)(a1 + 128);
    v68 = v72;
    ++*(_QWORD *)(a1 + 112);
    v69 = v67 + 1;
    v75 = v68;
    if ( 4 * v69 >= 3 * v66 )
    {
      v66 *= 2;
    }
    else if ( v66 - *(_DWORD *)(a1 + 132) - v69 > v66 >> 3 )
    {
LABEL_73:
      *(_DWORD *)(a1 + 128) = v69;
      if ( *v68 != -1 )
        --*(_DWORD *)(a1 + 132);
      *v68 = *v65;
      goto LABEL_69;
    }
    sub_A08C50(v6, v66);
    sub_22B31A0(v6, v65, &v75);
    v68 = v75;
    v69 = *(_DWORD *)(a1 + 128) + 1;
    goto LABEL_73;
  }
  v47 = 4 * v8;
  v9 = *(unsigned int *)(a1 + 136);
  if ( (unsigned int)(4 * v8) < 0x40 )
    v47 = 64;
  if ( (unsigned int)v9 <= v47 )
    goto LABEL_4;
  v48 = v8 - 1;
  if ( !v48 )
  {
    v53 = 512;
    v52 = 128;
    goto LABEL_51;
  }
  _BitScanReverse(&v48, v48);
  v49 = 1 << (33 - (v48 ^ 0x1F));
  if ( v49 < 64 )
    v49 = 64;
  if ( v49 != (_DWORD)v9 )
  {
    v50 = (4 * v49 / 3u + 1) | ((unsigned __int64)(4 * v49 / 3u + 1) >> 1);
    v51 = ((((v50 >> 2) | v50 | (((v50 >> 2) | v50) >> 4)) >> 8)
         | (v50 >> 2)
         | v50
         | (((v50 >> 2) | v50) >> 4)
         | (((((v50 >> 2) | v50 | (((v50 >> 2) | v50) >> 4)) >> 8) | (v50 >> 2) | v50 | (((v50 >> 2) | v50) >> 4)) >> 16))
        + 1;
    v52 = v51;
    v53 = 4 * v51;
LABEL_51:
    sub_C7D6A0(*(_QWORD *)(a1 + 120), 4 * v9, 4);
    *(_DWORD *)(a1 + 136) = v52;
    *(_QWORD *)(a1 + 120) = sub_C7D670(v53, 4);
  }
  sub_2C2BFC0(v6);
  v54 = *(_DWORD *)(a1 + 128);
  *(_DWORD *)(a1 + 152) = 0;
  if ( !v54 )
    goto LABEL_7;
  if ( (unsigned __int8)sub_22B31A0(v6, &v71, &v72) )
    goto LABEL_10;
  v56 = *(_DWORD *)(a1 + 136);
  v57 = *(_DWORD *)(a1 + 128);
  v58 = v72;
  ++*(_QWORD *)(a1 + 112);
  v59 = v57 + 1;
  v60 = 2 * v56;
  v75 = v58;
  if ( 4 * v59 >= 3 * v56 )
  {
    v56 *= 2;
    goto LABEL_80;
  }
  if ( v56 - *(_DWORD *)(a1 + 132) - v59 <= v56 >> 3 )
  {
LABEL_80:
    sub_A08C50(v6, v56);
    sub_22B31A0(v6, &v71, &v75);
    v58 = v75;
    v59 = *(_DWORD *)(a1 + 128) + 1;
  }
  *(_DWORD *)(a1 + 128) = v59;
  if ( *v58 != -1 )
    --*(_DWORD *)(a1 + 132);
  *v58 = v71;
  v61 = *(unsigned int *)(a1 + 152);
  v62 = v71;
  if ( v61 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 156) )
  {
    sub_C8D5F0(a1 + 144, (const void *)(a1 + 160), v61 + 1, 4u, v60, v55);
    v61 = *(unsigned int *)(a1 + 152);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 144) + 4 * v61) = v62;
  ++*(_DWORD *)(a1 + 152);
LABEL_10:
  if ( a2 == 1 )
    goto LABEL_41;
  v75 = (int *)a1;
  v76 = a2;
  v14 = sub_2BF3F10((_QWORD *)a1);
  v15 = sub_2BF04D0(v14);
  v18 = v15 + 112;
  if ( v15 + 112 == (*(_QWORD *)(v15 + 112) & 0xFFFFFFFFFFFFFFF8LL) )
  {
    if ( *(_DWORD *)(v15 + 88) != 1 )
      BUG();
    v15 = **(_QWORD **)(v15 + 80);
  }
  v19 = *(_QWORD *)(v15 + 120);
  if ( !v19 )
    BUG();
  v20 = *(unsigned int *)(v19 + 32);
  if ( !(_DWORD)v20 )
    BUG();
  v21 = *(_QWORD *)a1;
  v22 = *(_QWORD **)(*(_QWORD *)(**(_QWORD **)(v19 + 24) + 40LL) + 8LL);
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  v81 = v22;
  v23 = *v22;
  v83 = 0;
  v82 = v23;
  v84 = &v88;
  v72 = (int *)v74;
  v73 = 0x800000000LL;
  v85 = 8;
  v86 = 0;
  v87 = 1;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  sub_2C40060((__int64)&v72, v21, v20, v18, v16, v17);
  v28 = (unsigned __int64)v72;
  v29 = &v72[2 * (unsigned int)v73];
  if ( v72 != v29 )
  {
    do
    {
      v30 = *((_QWORD *)v29 - 1);
      v29 -= 2;
      sub_2C44BE0((__int64)&v75, v30, v24, v25, v26, v27);
    }
    while ( (int *)v28 != v29 );
  }
  v31 = 1;
  v32 = sub_2BF3F10((_QWORD *)a1);
  v33 = sub_2BF04D0(v32);
  v34 = sub_2BF05A0(v33);
  v35 = *(_QWORD *)(v33 + 120);
  v36 = v34;
  while ( v35 != v36 )
  {
    while ( 1 )
    {
      if ( !v35 )
        BUG();
      v39 = *(_BYTE *)(v35 - 16);
      if ( v39 == 32 )
      {
        v46 = sub_2C3F780((__int64)&v75, *(_QWORD *)(*(_QWORD *)(v35 + 24) + 8LL), a2 - 1);
        sub_2AAED30(v35 + 16, 1u, v46);
        goto LABEL_20;
      }
      v40 = (_QWORD **)(*(_QWORD *)(v35 - 8) & 0xFFFFFFFFFFFFFFF8LL);
      if ( (*(_QWORD *)(v35 - 8) & 4) != 0 )
        v40 = (_QWORD **)**v40;
      if ( v92 )
        break;
LABEL_26:
      if ( v39 == 34 )
        goto LABEL_19;
      v41 = v31++;
      sub_2C3FA00((__int64)&v75, v35 - 24, v41);
      v35 = *(_QWORD *)(v35 + 8);
      if ( v35 == v36 )
        goto LABEL_28;
    }
    v37 = (v92 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
    v38 = (_QWORD **)v90[9 * v37];
    if ( v38 != v40 )
    {
      v63 = 1;
      while ( v38 != (_QWORD **)-4096LL )
      {
        v37 = (v92 - 1) & (v63 + v37);
        v38 = (_QWORD **)v90[9 * v37];
        if ( v38 == v40 )
          goto LABEL_19;
        ++v63;
      }
      goto LABEL_26;
    }
LABEL_19:
    v31 = 1;
LABEL_20:
    v35 = *(_QWORD *)(v35 + 8);
  }
LABEL_28:
  sub_2C37F10((__int64 *)a1);
  if ( v72 != (int *)v74 )
    _libc_free((unsigned __int64)v72);
  v42 = v92;
  if ( v92 )
  {
    v43 = v90;
    v44 = &v90[9 * v92];
    do
    {
      if ( *v43 != -8192 && *v43 != -4096 )
      {
        v45 = v43[1];
        if ( (_QWORD *)v45 != v43 + 3 )
          _libc_free(v45);
      }
      v43 += 9;
    }
    while ( v44 != v43 );
    v42 = v92;
  }
  sub_C7D6A0((__int64)v90, 72 * v42, 8);
  if ( !v87 )
    _libc_free((unsigned __int64)v84);
  sub_C7D6A0(v78, 16LL * (unsigned int)v80, 8);
LABEL_41:
  sub_2C40480((__int64 *)a1);
}
