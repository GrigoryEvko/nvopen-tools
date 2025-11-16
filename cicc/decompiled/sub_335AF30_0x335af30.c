// Function: sub_335AF30
// Address: 0x335af30
//
void *__fastcall sub_335AF30(__int64 a1)
{
  char v2; // al
  __int64 v3; // r12
  void *v4; // rax
  void *v5; // rcx
  unsigned __int64 v6; // rdi
  __int64 v7; // r12
  void *v8; // rax
  void *v9; // rcx
  unsigned __int64 v10; // rdi
  unsigned int v11; // eax
  unsigned int v12; // eax
  unsigned int v13; // ecx
  __int64 v14; // rdx
  _QWORD *v15; // rax
  __int64 v16; // rdx
  _QWORD *i; // rdx
  __int64 v18; // rdi
  __int64 v19; // rdx
  void (__fastcall *v20)(__int64, __int64); // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  void (*v23)(void); // rax
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // r8
  __int64 v27; // rax
  __int64 v28; // rdi
  void (*v29)(void); // rax
  int v30; // eax
  _BYTE *v31; // r8
  __int64 v32; // rax
  const void *v33; // r15
  __int64 v34; // rcx
  __int64 v35; // r14
  __int64 v36; // rax
  char *v37; // r12
  _QWORD *v38; // rdi
  __int64 v39; // rcx
  bool (__fastcall *v40)(__int64); // rax
  unsigned __int64 *v41; // rax
  __int64 v42; // rdx
  void (__fastcall *v43)(__int64, __int64); // rcx
  __int64 v44; // r8
  __int64 v45; // r9
  __int64 v46; // r13
  __int64 v47; // rsi
  __int64 v48; // rdi
  __int64 (*v49)(); // rax
  __int64 v50; // r9
  _BYTE *v51; // rsi
  __int64 v52; // r8
  __int64 v53; // rdx
  void (__fastcall *v54)(__int64, __int64); // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // r9
  __int64 v58; // rdi
  __int64 v59; // r15
  __int64 v60; // rax
  __int64 k; // r13
  __int64 v62; // rax
  __int64 v63; // r13
  __int64 *v64; // rax
  __int64 *m; // rdi
  char v66; // si
  unsigned int *v67; // rdi
  __int64 (*v68)(); // rax
  unsigned int v69; // eax
  __int64 *v70; // rax
  __int64 *v71; // rdx
  __int64 *n; // rax
  __int64 v73; // rcx
  __int64 v74; // rsi
  _QWORD *v75; // rdi
  __int64 (*v76)(void); // rax
  void *result; // rax
  _BYTE *v78; // rdx
  _BYTE *v79; // rdi
  unsigned int v80; // eax
  __int64 v81; // r15
  int v82; // eax
  int v83; // eax
  unsigned int *v84; // rax
  __int64 v85; // rax
  unsigned int v86; // eax
  __int64 v87; // rsi
  int v88; // r15d
  __int64 v89; // rsi
  bool v90; // zf
  _QWORD *v91; // rax
  __int64 v92; // rdx
  _QWORD *j; // rdx
  unsigned int v94; // eax
  unsigned int v95; // r12d
  char v96; // al
  __int64 v97; // rdi
  __int64 v98; // rax
  _QWORD *v99; // rax
  __int64 v100; // rdx
  _QWORD *v101; // rdx
  __int64 v102; // [rsp+0h] [rbp-50h]
  __int64 v103; // [rsp+0h] [rbp-50h]
  __int64 v104; // [rsp+0h] [rbp-50h]
  __int64 v105; // [rsp+8h] [rbp-48h]
  __int64 v106[7]; // [rsp+18h] [rbp-38h] BYREF

  v2 = byte_5038F08;
  *(_DWORD *)(a1 + 680) = 0;
  *(_QWORD *)(a1 + 688) = 0;
  *(_DWORD *)(a1 + 684) = -((unsigned __int8)v2 ^ 1);
  v3 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 1);
  v4 = (void *)sub_2207820(8 * v3);
  v5 = v4;
  if ( v4 && v3 )
    v5 = memset(v4, 0, 8 * v3);
  v6 = *(_QWORD *)(a1 + 696);
  *(_QWORD *)(a1 + 696) = v5;
  if ( v6 )
    j_j___libc_free_0_0(v6);
  v7 = (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 24) + 16LL) + 1);
  v8 = (void *)sub_2207820(8 * v7);
  v9 = v8;
  if ( v8 && v7 )
    v9 = memset(v8, 0, 8 * v7);
  v10 = *(_QWORD *)(a1 + 704);
  *(_QWORD *)(a1 + 704) = v9;
  if ( v10 )
    j_j___libc_free_0_0(v10);
  v11 = *(_DWORD *)(a1 + 1216);
  ++*(_QWORD *)(a1 + 1208);
  v12 = v11 >> 1;
  if ( v12 )
  {
    if ( (*(_BYTE *)(a1 + 1216) & 1) == 0 )
    {
      v13 = 4 * v12;
      goto LABEL_14;
    }
LABEL_118:
    v15 = (_QWORD *)(a1 + 1224);
    v16 = 32;
    goto LABEL_17;
  }
  if ( !*(_DWORD *)(a1 + 1220) )
    goto LABEL_20;
  v13 = 0;
  if ( (*(_BYTE *)(a1 + 1216) & 1) != 0 )
    goto LABEL_118;
LABEL_14:
  v14 = *(unsigned int *)(a1 + 1232);
  if ( (unsigned int)v14 <= v13 || (unsigned int)v14 <= 0x40 )
  {
    v15 = *(_QWORD **)(a1 + 1224);
    v16 = 2 * v14;
LABEL_17:
    for ( i = &v15[v16]; i != v15; v15 += 2 )
      *v15 = -4096;
    *(_QWORD *)(a1 + 1216) &= 1uLL;
    goto LABEL_20;
  }
  if ( !v12 || (v94 = v12 - 1) == 0 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 1224), 16 * v14, 8);
    *(_BYTE *)(a1 + 1216) |= 1u;
    goto LABEL_128;
  }
  _BitScanReverse(&v94, v94);
  v95 = 1 << (33 - (v94 ^ 0x1F));
  if ( v95 - 17 <= 0x2E )
  {
    v95 = 64;
    sub_C7D6A0(*(_QWORD *)(a1 + 1224), 16 * v14, 8);
    v96 = *(_BYTE *)(a1 + 1216);
    v97 = 1024;
    goto LABEL_139;
  }
  if ( (_DWORD)v14 != v95 )
  {
    sub_C7D6A0(*(_QWORD *)(a1 + 1224), 16 * v14, 8);
    v96 = *(_BYTE *)(a1 + 1216) | 1;
    *(_BYTE *)(a1 + 1216) = v96;
    if ( v95 <= 0x10 )
    {
LABEL_128:
      v90 = (*(_QWORD *)(a1 + 1216) & 1LL) == 0;
      *(_QWORD *)(a1 + 1216) &= 1uLL;
      if ( v90 )
      {
        v91 = *(_QWORD **)(a1 + 1224);
        v92 = 2LL * *(unsigned int *)(a1 + 1232);
      }
      else
      {
        v91 = (_QWORD *)(a1 + 1224);
        v92 = 32;
      }
      for ( j = &v91[v92]; j != v91; v91 += 2 )
      {
        if ( v91 )
          *v91 = -4096;
      }
      goto LABEL_20;
    }
    v97 = 16LL * v95;
LABEL_139:
    *(_BYTE *)(a1 + 1216) = v96 & 0xFE;
    v98 = sub_C7D670(v97, 8);
    *(_DWORD *)(a1 + 1232) = v95;
    *(_QWORD *)(a1 + 1224) = v98;
    goto LABEL_128;
  }
  v90 = (*(_QWORD *)(a1 + 1216) & 1LL) == 0;
  *(_QWORD *)(a1 + 1216) &= 1uLL;
  if ( v90 )
  {
    v99 = *(_QWORD **)(a1 + 1224);
    v100 = 2 * v14;
  }
  else
  {
    v99 = (_QWORD *)(a1 + 1224);
    v100 = 32;
  }
  v101 = &v99[v100];
  do
  {
    if ( v99 )
      *v99 = -4096;
    v99 += 2;
  }
  while ( v101 != v99 );
LABEL_20:
  sub_3360940(a1);
  v18 = *(_QWORD *)(a1 + 640);
  *(_BYTE *)(a1 + 808) = 1;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 32LL))(v18, a1 + 48);
  v23 = *(void (**)(void))(**(_QWORD **)(a1 + 672) + 32LL);
  if ( v23 != nullsub_1618 )
    v23();
  v24 = a1 + 328;
  sub_3356F00(a1, a1 + 328, v19, v20, v21, v22);
  v27 = *(_QWORD *)(a1 + 48);
  if ( *(_QWORD *)(a1 + 56) == v27 )
  {
    v105 = a1 + 608;
  }
  else
  {
    v24 = v27 + ((__int64)*(int *)(*(_QWORD *)(*(_QWORD *)(a1 + 592) + 384LL) + 36LL) << 8);
    *(_BYTE *)(v24 + 249) |= 2u;
    v28 = *(_QWORD *)(a1 + 640);
    v29 = *(void (**)(void))(*(_QWORD *)v28 + 88LL);
    if ( (char *)v29 == (char *)sub_33549A0 )
    {
      v106[0] = v24;
      v30 = *(_DWORD *)(v28 + 40) + 1;
      *(_DWORD *)(v28 + 40) = v30;
      *(_DWORD *)(v24 + 204) = v30;
      v31 = *(_BYTE **)(v28 + 24);
      if ( v31 == *(_BYTE **)(v28 + 32) )
      {
        v24 = *(_QWORD *)(v28 + 24);
        sub_2ECAD30(v28 + 16, v31, v106);
      }
      else
      {
        if ( v31 )
        {
          *(_QWORD *)v31 = v24;
          v31 = *(_BYTE **)(v28 + 24);
        }
        v26 = (__int64)(v31 + 8);
        *(_QWORD *)(v28 + 24) = v26;
      }
    }
    else
    {
      v29();
    }
    v105 = a1 + 608;
    v32 = *(_QWORD *)(a1 + 56) - *(_QWORD *)(a1 + 48);
    v25 = v32 >> 8;
    if ( v32 < 0 )
      sub_4262D8((__int64)"vector::reserve");
    v33 = *(const void **)(a1 + 608);
    if ( v25 > (unsigned __int64)((__int64)(*(_QWORD *)(a1 + 624) - (_QWORD)v33) >> 3) )
    {
      v34 = 8 * v25;
      v35 = *(_QWORD *)(a1 + 616) - (_QWORD)v33;
      if ( v25 )
      {
        v102 = 8 * v25;
        v36 = sub_22077B0(8 * v25);
        v33 = *(const void **)(a1 + 608);
        v34 = v102;
        v37 = (char *)v36;
        v25 = *(_QWORD *)(a1 + 616) - (_QWORD)v33;
        if ( v25 <= 0 )
        {
LABEL_32:
          if ( !v33 )
          {
LABEL_33:
            *(_QWORD *)(a1 + 608) = v37;
            *(_QWORD *)(a1 + 616) = &v37[v35];
            *(_QWORD *)(a1 + 624) = &v37[v34];
            goto LABEL_34;
          }
          v24 = *(_QWORD *)(a1 + 624) - (_QWORD)v33;
LABEL_122:
          v104 = v34;
          j_j___libc_free_0((unsigned __int64)v33);
          v34 = v104;
          goto LABEL_33;
        }
      }
      else
      {
        v25 = *(_QWORD *)(a1 + 616) - (_QWORD)v33;
        v37 = 0;
        if ( v35 <= 0 )
          goto LABEL_32;
      }
      v103 = v34;
      memmove(v37, v33, v25);
      v34 = v103;
      v24 = *(_QWORD *)(a1 + 624) - (_QWORD)v33;
      goto LABEL_122;
    }
  }
LABEL_34:
  v38 = *(_QWORD **)(a1 + 640);
  v39 = (__int64)v106;
  v40 = *(bool (__fastcall **)(__int64))(*v38 + 64LL);
LABEL_35:
  if ( v40 == sub_3351550 )
    goto LABEL_36;
  while ( 2 )
  {
    if ( ((unsigned __int8 (*)(void))v40)() )
    {
      v26 = *(unsigned int *)(a1 + 720);
      if ( !(_DWORD)v26 )
        goto LABEL_80;
    }
LABEL_38:
    v41 = sub_33583E0((unsigned __int64 *)a1, v24, v25, v39, v26);
    v46 = (__int64)v41;
    if ( byte_5038F08 )
      goto LABEL_45;
    if ( (*((_BYTE *)v41 + 254) & 2) == 0 )
      sub_2F8F770((__int64)v41, (_QWORD *)v24, v42, (__int64)v43, v44, v45);
    v47 = *(unsigned int *)(v46 + 244);
    if ( (unsigned int)v47 > *(_DWORD *)(a1 + 680) )
      sub_3354BF0(a1, v47, v42, v43, v44, v45);
    if ( (*(_BYTE *)(v46 + 248) & 2) != 0 )
      goto LABEL_45;
    v48 = *(_QWORD *)(a1 + 672);
    v49 = *(__int64 (**)())(*(_QWORD *)v48 + 24LL);
    if ( v49 == sub_2EC0B50 )
      goto LABEL_45;
    v88 = 0;
    do
    {
      if ( !((unsigned int (__fastcall *)(__int64, __int64, _QWORD))v49)(v48, v46, (unsigned int)-v88) )
        break;
      v48 = *(_QWORD *)(a1 + 672);
      ++v88;
      v49 = *(__int64 (**)())(*(_QWORD *)v48 + 24LL);
    }
    while ( v49 != sub_2EC0B50 );
    v50 = *(unsigned int *)(a1 + 680);
    v89 = (unsigned int)(v88 + v50);
    if ( (unsigned int)v50 < (unsigned int)v89 )
    {
      sub_3354BF0(a1, v89, v42, v43, v44, v50);
LABEL_45:
      v50 = *(unsigned int *)(a1 + 680);
    }
    v106[0] = v46;
    sub_2F8F8C0(v46, (_QWORD *)(unsigned int)v50, v42, (__int64)v43, v44, v50);
    sub_3351E00(a1, v106[0]);
    v51 = *(_BYTE **)(a1 + 616);
    if ( v51 == *(_BYTE **)(a1 + 624) )
    {
      sub_2ECAD30(v105, v51, v106);
      v52 = v106[0];
    }
    else
    {
      v52 = v106[0];
      if ( v51 )
      {
        *(_QWORD *)v51 = v106[0];
        v51 = *(_BYTE **)(a1 + 616);
      }
      *(_QWORD *)(a1 + 616) = v51 + 8;
    }
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 640) + 120LL))(*(_QWORD *)(a1 + 640), v52);
    if ( !*(_DWORD *)(*(_QWORD *)(a1 + 672) + 8LL) && (unsigned int)qword_5038648 <= 1 )
    {
      v86 = *(_DWORD *)(a1 + 680);
      v87 = v86 + 1;
      if ( v86 < (unsigned int)v87 )
        sub_3354BF0(a1, v87, v53, v54, v55, v56);
    }
    sub_3356F00(a1, v106[0], v53, v54, v55, v56);
    v39 = v106[0];
    v58 = *(_QWORD *)(a1 + 696);
    v59 = *(_QWORD *)(v106[0] + 120);
    v60 = 16LL * *(unsigned int *)(v106[0] + 128);
    for ( k = v59 + v60; k != v59; v59 += 16 )
    {
      if ( (*(_BYTE *)v59 & 6) == 0 )
      {
        v62 = *(unsigned int *)(v59 + 8);
        if ( (_DWORD)v62 )
        {
          if ( *(_QWORD *)(v58 + 8 * v62) == v39 )
          {
            --*(_DWORD *)(a1 + 692);
            *(_QWORD *)(v58 + 8LL * *(unsigned int *)(v59 + 8)) = 0;
            *(_QWORD *)(*(_QWORD *)(a1 + 704) + 8LL * *(unsigned int *)(v59 + 8)) = 0;
            sub_3354CB0(a1, *(_DWORD *)(v59 + 8));
            v58 = *(_QWORD *)(a1 + 696);
            v39 = v106[0];
          }
        }
      }
    }
    v63 = *(unsigned int *)(*(_QWORD *)(a1 + 24) + 16LL);
    if ( *(_QWORD *)(v58 + 8 * v63) == v39 )
    {
      v81 = *(_QWORD *)v39;
      do
      {
        if ( !v81 )
          break;
        v82 = *(_DWORD *)(v81 + 24);
        if ( v82 < 0 && ~v82 == *(_DWORD *)(*(_QWORD *)(a1 + 16) + 64LL) )
        {
          v85 = *(_QWORD *)(a1 + 696);
          --*(_DWORD *)(a1 + 692);
          *(_QWORD *)(v85 + 8 * v63) = 0;
          *(_QWORD *)(*(_QWORD *)(a1 + 704) + 8 * v63) = 0;
          sub_3354CB0(a1, v63);
        }
        v83 = *(_DWORD *)(v81 + 64);
        if ( !v83 )
          break;
        v84 = (unsigned int *)(*(_QWORD *)(v81 + 40) + 40LL * (unsigned int)(v83 - 1));
        v81 = *(_QWORD *)v84;
      }
      while ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v84 + 48LL) + 16LL * v84[2]) == 262 );
      v39 = v106[0];
    }
    if ( (*(_BYTE *)(v39 + 248) & 1) != 0 )
    {
      v64 = *(__int64 **)(v39 + 40);
      for ( m = &v64[2 * *(unsigned int *)(v39 + 48)]; m != v64; v64 += 2 )
      {
        v25 = *v64;
        if ( (*v64 & 6) == 0 )
        {
          v25 &= 0xFFFFFFFFFFFFFFF8LL;
          v66 = *(_BYTE *)(v25 + 248);
          if ( (v66 & 1) != 0 )
            *(_BYTE *)(v25 + 248) = v66 & 0xFE;
        }
      }
    }
    *(_BYTE *)(v39 + 249) |= 4u;
    v67 = *(unsigned int **)(a1 + 672);
    v24 = v67[2];
    if ( (_DWORD)v24 )
    {
      if ( !*(_QWORD *)v39 || *(int *)(*(_QWORD *)v39 + 24LL) >= 0 )
      {
        v68 = *(__int64 (**)())(*(_QWORD *)v67 + 16LL);
        if ( v68 == sub_2F39220 )
          goto LABEL_69;
        goto LABEL_104;
      }
    }
    else
    {
      if ( (unsigned int)qword_5038648 <= 1 )
        goto LABEL_69;
      if ( !*(_QWORD *)v39 || (v25 = *(unsigned int *)(*(_QWORD *)v39 + 24LL), (int)v25 >= 0) )
      {
LABEL_91:
        if ( *(_DWORD *)(a1 + 688) != (_DWORD)qword_5038648 )
          goto LABEL_69;
        goto LABEL_92;
      }
    }
    ++*(_DWORD *)(a1 + 688);
    v39 = v67[2];
    if ( !(_DWORD)v39 )
      goto LABEL_91;
    v68 = *(__int64 (**)())(*(_QWORD *)v67 + 16LL);
    if ( v68 == sub_2F39220 )
      goto LABEL_69;
LABEL_104:
    if ( !(unsigned __int8)v68() )
    {
      if ( *(_DWORD *)(*(_QWORD *)(a1 + 672) + 8LL) )
        goto LABEL_69;
      goto LABEL_91;
    }
LABEL_92:
    v80 = *(_DWORD *)(a1 + 680);
    v24 = v80 + 1;
    if ( v80 < (unsigned int)v24 )
      goto LABEL_75;
LABEL_69:
    while ( 1 )
    {
      v38 = *(_QWORD **)(a1 + 640);
      v40 = *(bool (__fastcall **)(__int64))(*v38 + 64LL);
      if ( v40 != sub_3351550 )
        break;
      v39 = v38[2];
      if ( v38[3] != v39 )
        goto LABEL_35;
LABEL_71:
      if ( *(_QWORD *)(a1 + 656) == *(_QWORD *)(a1 + 648) )
        goto LABEL_77;
      v69 = *(_DWORD *)(a1 + 680);
      v25 = *(unsigned int *)(a1 + 684);
      v24 = v69 + 1;
      if ( (unsigned int)v24 < (unsigned int)v25 )
        v24 = (unsigned int)v25;
      if ( v69 < (unsigned int)v24 )
LABEL_75:
        sub_3354BF0(a1, v24, v25, (void (__fastcall *)(__int64, __int64))v39, v26, v57);
    }
    if ( ((unsigned __int8 (*)(void))v40)() )
      goto LABEL_71;
LABEL_77:
    v38 = *(_QWORD **)(a1 + 640);
    v40 = *(bool (__fastcall **)(__int64))(*v38 + 64LL);
    if ( v40 != sub_3351550 )
      continue;
    break;
  }
LABEL_36:
  if ( v38[3] != v38[2] )
    goto LABEL_38;
  v26 = *(unsigned int *)(a1 + 720);
  if ( (_DWORD)v26 )
    goto LABEL_38;
LABEL_80:
  v70 = *(__int64 **)(a1 + 616);
  v71 = *(__int64 **)(a1 + 608);
  if ( v70 != v71 )
  {
    for ( n = v70 - 1; n > v71; n[1] = v73 )
    {
      v73 = *v71;
      v74 = *n;
      ++v71;
      --n;
      *(v71 - 1) = v74;
    }
  }
  v75 = *(_QWORD **)(a1 + 640);
  v76 = *(__int64 (**)(void))(*v75 + 56LL);
  if ( (char *)v76 != (char *)sub_3351510 )
    return (void *)v76();
  v75[6] = 0;
  result = (void *)v75[12];
  if ( result != (void *)v75[13] )
    v75[13] = result;
  v78 = (_BYTE *)v75[16];
  v79 = (_BYTE *)v75[15];
  if ( v78 != v79 )
    return memset(v79, 0, v78 - v79);
  return result;
}
