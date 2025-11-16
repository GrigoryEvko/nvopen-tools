// Function: sub_D58ED0
// Address: 0xd58ed0
//
__int64 __fastcall sub_D58ED0(__int64 a1, unsigned __int64 a2, __m128i a3)
{
  __int64 *v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 i; // rax
  __int64 v12; // rbx
  __int64 v13; // rbx
  __int64 j; // r13
  __int64 v15; // rdi
  __int64 *v16; // r13
  __int64 v17; // r12
  unsigned int v18; // ebx
  __int64 v19; // rdi
  __int64 (*v20)(); // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rdx
  int v26; // r8d
  const char *v27; // r8
  size_t v28; // r9
  size_t v29; // rcx
  const char *v30; // r9
  __int64 v31; // r12
  const char *v32; // rax
  size_t v33; // rdx
  __int64 v34; // r13
  _QWORD *v35; // rax
  __int64 v36; // r13
  __int64 (__fastcall *v37)(__int64, __int64, __m128i); // rax
  bool v38; // r14
  _QWORD *v39; // rax
  _QWORD *v40; // rcx
  char *v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rax
  __int64 v44; // r13
  __int64 v45; // rax
  size_t v46; // rdx
  unsigned int v47; // eax
  unsigned int v48; // edx
  char v49; // al
  __int64 v50; // rdi
  size_t v51; // rdx
  unsigned int k; // r13d
  __int64 v53; // rdx
  __int64 v54; // rax
  int v55; // r13d
  unsigned int v56; // ebx
  __int64 v57; // rax
  __int64 (*v58)(); // rax
  __int64 v59; // rbx
  __int64 v60; // r12
  __int64 v61; // rdi
  __int64 v62; // r8
  __int64 v63; // r12
  __int64 v64; // rbx
  _QWORD *v65; // rdi
  __int64 *v67; // rdx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 *v70; // [rsp+8h] [rbp-C8h]
  __int64 *v71; // [rsp+10h] [rbp-C0h]
  unsigned int v72; // [rsp+1Ch] [rbp-B4h]
  unsigned int v74; // [rsp+28h] [rbp-A8h]
  unsigned __int8 v75; // [rsp+2Eh] [rbp-A2h]
  char v76; // [rsp+2Fh] [rbp-A1h]
  __int64 v77; // [rsp+30h] [rbp-A0h]
  unsigned int v78; // [rsp+30h] [rbp-A0h]
  __int64 v79; // [rsp+38h] [rbp-98h]
  __int64 *v80; // [rsp+40h] [rbp-90h]
  __int64 *v81; // [rsp+48h] [rbp-88h]
  unsigned int v82; // [rsp+48h] [rbp-88h]
  __int64 v83; // [rsp+50h] [rbp-80h] BYREF
  __int64 v84; // [rsp+58h] [rbp-78h]
  __int64 v85; // [rsp+60h] [rbp-70h]
  _QWORD v86[12]; // [rsp+70h] [rbp-60h] BYREF

  v3 = *(__int64 **)(a1 + 8);
  v4 = *v3;
  v5 = v3[1];
  if ( v4 == v5 )
LABEL_82:
    BUG();
  while ( *(_UNKNOWN **)v4 != &unk_4F875EC )
  {
    v4 += 16;
    if ( v5 == v4 )
      goto LABEL_82;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v4 + 8) + 104LL))(*(_QWORD *)(v4 + 8), &unk_4F875EC);
  v8 = *(_QWORD *)(a1 + 184);
  v9 = a1 + 336;
  v71 = (__int64 *)v7;
  *(_QWORD *)(a1 + 648) = v7 + 176;
  v10 = *(_QWORD *)(v8 + 8);
  v70 = *(__int64 **)(a2 + 40);
  for ( i = *(_QWORD *)(v8 + 16); v10 != i; *(_QWORD *)(v9 - 8) = v12 + 208 )
  {
    v12 = *(_QWORD *)(i - 8);
    i -= 8;
    v9 += 8;
  }
  v13 = v71[26];
  for ( j = v71[27]; j != v13; j -= 8 )
  {
    v15 = *(_QWORD *)(j - 8);
    sub_D58D90(v15, (__int64 *)(a1 + 568));
  }
  v16 = *(__int64 **)(a1 + 584);
  v80 = *(__int64 **)(a1 + 616);
  if ( v80 == v16 )
    return 0;
  v75 = 0;
  v81 = *(__int64 **)(a1 + 600);
  v79 = *(_QWORD *)(a1 + 608);
  do
  {
    v17 = *v16;
    if ( *(_DWORD *)(a1 + 200) )
    {
      v18 = 0;
      do
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v18);
          v20 = *(__int64 (**)())(*(_QWORD *)v19 + 152LL);
          if ( v20 != sub_CE11A0 )
            break;
          if ( ++v18 >= *(_DWORD *)(a1 + 200) )
            goto LABEL_16;
        }
        ++v18;
        v75 |= ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64))v20)(v19, v17, a1, v9);
      }
      while ( v18 < *(_DWORD *)(a1 + 200) );
    }
LABEL_16:
    if ( ++v16 == v81 )
    {
      v16 = *(__int64 **)(v79 + 8);
      v79 += 8;
      v81 = v16 + 64;
    }
  }
  while ( v80 != v16 );
  v83 = 0;
  v85 = 0x1000000000LL;
  v84 = 0;
  v21 = sub_B6F970(*v70);
  v22 = (__int64)"size-info";
  v74 = 0;
  v76 = (*(__int64 (__fastcall **)(__int64, const char *, __int64))(*(_QWORD *)v21 + 24LL))(v21, "size-info", 9);
  if ( v76 )
  {
    v22 = (__int64)v70;
    v72 = sub_B806A0(a1 + 176, (__int64)v70, (__int64)&v83);
    v74 = sub_B2BED0(a2);
  }
  v23 = *(_QWORD *)(a1 + 616);
  if ( v23 == *(_QWORD *)(a1 + 584) )
    goto LABEL_62;
  do
  {
    v24 = *(_QWORD *)(a1 + 624);
    *(_BYTE *)(a1 + 664) = 0;
    if ( v24 == v23 )
      v23 = *(_QWORD *)(*(_QWORD *)(a1 + 640) - 8LL) + 512LL;
    v25 = *(_QWORD *)(v23 - 8);
    v26 = *(_DWORD *)(a1 + 200);
    *(_QWORD *)(a1 + 656) = v25;
    if ( !v26 )
      goto LABEL_59;
    v82 = 0;
    while ( 1 )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8LL * v82);
      v32 = sub_BD5D20(**(_QWORD **)(v25 + 32));
      sub_B817B0(a1 + 176, v31, 0, 6, v32, v33);
      sub_B86470(a1 + 176, (__int64 *)v31);
      sub_B89740(a1 + 176, (__int64 *)v31);
      v34 = **(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL);
      sub_C85EE0(v86);
      v86[2] = v31;
      v86[3] = v34;
      v86[0] = &unk_49DA748;
      v86[4] = 0;
      v35 = sub_BC4450((__int64 *)v31);
      v36 = (__int64)v35;
      if ( v35 )
        sub_C9E250((__int64)v35);
      v37 = *(__int64 (__fastcall **)(__int64, __int64, __m128i))(*(_QWORD *)v31 + 144LL);
      if ( v37 == sub_D57490 )
      {
        v77 = *(_QWORD *)(a1 + 656);
        v38 = 0;
        v39 = sub_D573D0(*(_QWORD **)(v77 + 32), *(_QWORD *)(v77 + 40));
        if ( v40 != v39 )
        {
          v41 = (char *)sub_BD5D20(*(_QWORD *)(*v39 + 72LL));
          v38 = sub_BC63A0(v41, v42);
          if ( v38 )
          {
            v38 = 0;
            sub_D4BD90(v77, *(char **)(v31 + 176), v31 + 184, a3);
          }
        }
      }
      else
      {
        v49 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v37)(v31, *(_QWORD *)(a1 + 656), a1);
        v75 |= v49;
        v38 = v49;
      }
      if ( v76 )
      {
        v47 = sub_B2BED0(a2);
        if ( v47 != v74 )
        {
          v78 = v47;
          sub_B82CC0(a1 + 176, v31, (__int64)v70, v47 - (unsigned __int64)v74, v72, (__int64)&v83, a2);
          v48 = v72 - v74;
          v74 = v78;
          v72 = v78 + v48;
        }
      }
      if ( v36 )
        sub_C9E2A0(v36);
      v86[0] = &unk_49DA748;
      nullsub_162();
      if ( v38 )
      {
        v27 = "<deleted loop>";
        v28 = 14;
        if ( !*(_BYTE *)(a1 + 664) )
        {
          v28 = 14;
          v27 = "<unnamed loop>";
          v50 = **(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL);
          if ( v50 )
          {
            if ( (*(_BYTE *)(v50 + 7) & 0x10) != 0 )
            {
              v27 = sub_BD5D20(v50);
              v28 = v51;
            }
          }
        }
        sub_B817B0(a1 + 176, v31, 1, 6, v27, v28);
        sub_B865A0(a1 + 176, (__int64 *)v31);
        if ( *(_BYTE *)(a1 + 664) )
        {
LABEL_28:
          sub_B887D0(a1 + 176, (__int64 *)v31);
          goto LABEL_29;
        }
      }
      else
      {
        sub_B865A0(a1 + 176, (__int64 *)v31);
        if ( *(_BYTE *)(a1 + 664) )
          goto LABEL_29;
      }
      v43 = sub_BC4450(v71);
      v44 = (__int64)v43;
      if ( v43 )
      {
        sub_C9E250((__int64)v43);
        nullsub_188();
        sub_C9E2A0(v44);
      }
      else
      {
        nullsub_188();
      }
      nullsub_76();
      v45 = sub_B2BE50(a2);
      sub_B6EAA0(v45);
      if ( v38 )
        goto LABEL_28;
LABEL_29:
      sub_B87180(a1 + 176, v31);
      v29 = 9;
      v30 = "<deleted>";
      if ( !*(_BYTE *)(a1 + 664) )
      {
        v30 = sub_BD5D20(**(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL));
        v29 = v46;
      }
      v22 = v31;
      sub_B81BF0(a1 + 176, v31, v30, v29, 6);
      if ( *(_BYTE *)(a1 + 664) )
        break;
      if ( ++v82 >= *(_DWORD *)(a1 + 200) )
        goto LABEL_58;
      v25 = *(_QWORD *)(a1 + 656);
    }
    for ( k = 0; k < *(_DWORD *)(a1 + 200); ++k )
    {
      v53 = k;
      v22 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + 8 * v53);
      sub_B81AB0(a1 + 176, (_QWORD *)v22, "<deleted>", 9u, 6);
    }
LABEL_58:
    v24 = *(_QWORD *)(a1 + 624);
LABEL_59:
    v54 = *(_QWORD *)(a1 + 616);
    if ( v54 == v24 )
    {
      v22 = 512;
      j_j___libc_free_0(v24, 512);
      v67 = (__int64 *)(*(_QWORD *)(a1 + 640) - 8LL);
      *(_QWORD *)(a1 + 640) = v67;
      v68 = *v67;
      v69 = *v67 + 512;
      *(_QWORD *)(a1 + 624) = v68;
      v23 = v68 + 504;
      *(_QWORD *)(a1 + 632) = v69;
    }
    else
    {
      v23 = v54 - 8;
    }
    *(_QWORD *)(a1 + 616) = v23;
  }
  while ( v23 != *(_QWORD *)(a1 + 584) );
LABEL_62:
  if ( *(_DWORD *)(a1 + 200) )
  {
    v55 = v75;
    v56 = 0;
    do
    {
      while ( 1 )
      {
        v58 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 192) + 8LL * v56) + 160LL);
        if ( v58 != sub_CE11B0 )
          break;
        v57 = *(unsigned int *)(a1 + 200);
        if ( ++v56 >= (unsigned int)v57 )
          goto LABEL_67;
      }
      ++v56;
      v55 |= v58();
      v57 = *(unsigned int *)(a1 + 200);
    }
    while ( v56 < (unsigned int)v57 );
LABEL_67:
    v75 = v55;
    if ( (_BYTE)v55 && (_DWORD)v57 )
    {
      v59 = 8 * v57;
      v60 = 0;
      do
      {
        v22 = a2;
        v61 = *(_QWORD *)(*(_QWORD *)(a1 + 192) + v60);
        v60 += 8;
        sub_B7FEB0(v61, a2);
      }
      while ( v59 != v60 );
    }
  }
  v62 = v83;
  if ( HIDWORD(v84) && (_DWORD)v84 )
  {
    v63 = 8LL * (unsigned int)v84;
    v64 = 0;
    do
    {
      v65 = *(_QWORD **)(v62 + v64);
      if ( v65 != (_QWORD *)-8LL && v65 )
      {
        v22 = *v65 + 17LL;
        sub_C7D6A0((__int64)v65, v22, 8);
        v62 = v83;
      }
      v64 += 8;
    }
    while ( v63 != v64 );
  }
  _libc_free(v62, v22);
  return v75;
}
