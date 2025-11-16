// Function: sub_144B620
// Address: 0x144b620
//
__int64 __fastcall sub_144B620(__int64 a1)
{
  __int64 *v1; // rdx
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 i; // rsi
  __int64 v11; // rdx
  __int64 *v12; // r13
  __int64 v13; // r12
  unsigned int v14; // ebx
  __int64 v15; // rdi
  __int64 (*v16)(); // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  int v19; // edx
  __int64 v20; // rsi
  char v21; // al
  __int64 v22; // rdx
  __int64 v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // r8
  unsigned int v26; // eax
  _QWORD *v27; // r12
  __int64 v28; // rbx
  __int64 v29; // rax
  __int64 v30; // rbx
  char v31; // al
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // rdx
  unsigned int v35; // ebx
  __int64 v36; // rdx
  __int64 v37; // rax
  _QWORD *v38; // rdi
  int v39; // r13d
  unsigned int v40; // ebx
  __int64 (*v41)(); // rax
  _QWORD *v43; // rax
  __int64 *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rbx
  _QWORD *v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // [rsp+10h] [rbp-90h]
  unsigned __int8 v53; // [rsp+1Fh] [rbp-81h]
  __int64 *v54; // [rsp+20h] [rbp-80h]
  __int64 *v55; // [rsp+28h] [rbp-78h]
  unsigned int v56; // [rsp+28h] [rbp-78h]
  __int64 v57; // [rsp+38h] [rbp-68h] BYREF
  __m128i v58; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v59[10]; // [rsp+50h] [rbp-50h] BYREF

  v1 = *(__int64 **)(a1 + 8);
  v2 = *v1;
  v3 = v1[1];
  if ( v2 == v3 )
LABEL_87:
    BUG();
  while ( *(_UNKNOWN **)v2 != &unk_4F9A04C )
  {
    v2 += 16;
    if ( v3 == v2 )
      goto LABEL_87;
  }
  v5 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v2 + 8) + 104LL))(*(_QWORD *)(v2 + 8), &unk_4F9A04C);
  v6 = *(_QWORD *)(a1 + 176);
  v7 = a1 + 328;
  v8 = v5;
  *(_QWORD *)(a1 + 656) = v5 + 160;
  v9 = *(_QWORD *)(v6 + 16);
  for ( i = *(_QWORD *)(v6 + 8); i != v9; *(_QWORD *)(v7 - 8) = v11 + 224 )
  {
    v11 = *(_QWORD *)(v9 - 8);
    v9 -= 8;
    v7 += 8;
  }
  sub_144B4E0(*(_QWORD *)(v8 + 192), (__int64 *)(a1 + 568));
  v12 = *(__int64 **)(a1 + 584);
  v54 = *(__int64 **)(a1 + 616);
  if ( v12 == v54 )
    return 0;
  v53 = 0;
  v55 = *(__int64 **)(a1 + 600);
  v52 = *(_QWORD *)(a1 + 608);
  do
  {
    v13 = *v12;
    if ( *(_DWORD *)(a1 + 192) )
    {
      v14 = 0;
      do
      {
        while ( 1 )
        {
          v15 = *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8LL * v14);
          v16 = *(__int64 (**)())(*(_QWORD *)v15 + 152LL);
          if ( v16 != sub_144A6E0 )
            break;
          if ( ++v14 >= *(_DWORD *)(a1 + 192) )
            goto LABEL_14;
        }
        ++v14;
        v53 |= ((__int64 (__fastcall *)(__int64, __int64, __int64))v16)(v15, v13, a1);
      }
      while ( v14 < *(_DWORD *)(a1 + 192) );
    }
LABEL_14:
    if ( v55 == ++v12 )
    {
      v12 = *(__int64 **)(v52 + 8);
      v52 += 8;
      v55 = v12 + 64;
    }
  }
  while ( v54 != v12 );
  v17 = *(_QWORD *)(a1 + 616);
  if ( *(_QWORD *)(a1 + 584) == v17 )
    goto LABEL_67;
  do
  {
    v18 = *(_QWORD *)(a1 + 624);
    if ( v17 == v18 )
      v17 = *(_QWORD *)(*(_QWORD *)(a1 + 640) - 8LL) + 512LL;
    v19 = *(_DWORD *)(a1 + 192);
    *(_QWORD *)(a1 + 664) = *(_QWORD *)(v17 - 8);
    *(_WORD *)(a1 + 648) = 0;
    if ( !v19 )
      goto LABEL_61;
    v56 = 0;
    while ( 1 )
    {
      v27 = *(_QWORD **)(*(_QWORD *)(a1 + 184) + 8LL * v56);
      if ( (unsigned __int8)sub_160E750(a1 + 160) )
      {
        sub_1442FC0(&v58, *(__int64 **)(a1 + 664));
        sub_160F160(a1 + 160, v27, 0, 6, v58.m128i_i64[0], v58.m128i_i64[1]);
        if ( (_QWORD *)v58.m128i_i64[0] != v59 )
          j_j___libc_free_0(v58.m128i_i64[0], v59[0] + 1LL);
        sub_1615D60(a1 + 160, v27);
      }
      sub_1614C80(a1 + 160, v27);
      v28 = **(_QWORD **)(a1 + 664);
      sub_16C6860(&v58);
      v59[0] = v27;
      v59[2] = 0;
      v59[1] = v28 & 0xFFFFFFFFFFFFFFF8LL;
      v58.m128i_i64[0] = (__int64)&unk_49ED7C0;
      v29 = sub_1612E30(v27);
      v30 = v29;
      if ( v29 )
      {
        sub_16D7910(v29);
        sub_1403F30(&v57, v27, *(_QWORD *)(a1 + 168));
        v20 = *(_QWORD *)(a1 + 664);
        v21 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v27 + 144LL))(v27, v20, a1);
        v23 = v57;
        v53 |= v21;
        if ( !v57 )
          goto LABEL_26;
      }
      else
      {
        sub_1403F30(&v57, v27, *(_QWORD *)(a1 + 168));
        v20 = *(_QWORD *)(a1 + 664);
        v31 = (*(__int64 (__fastcall **)(_QWORD *, __int64, __int64))(*v27 + 144LL))(v27, v20, a1);
        v23 = v57;
        v53 |= v31;
        if ( !v57 )
          goto LABEL_27;
      }
      if ( v53 )
      {
        v20 = 2;
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v23 + 56LL))(v23, 2);
        v23 = v57;
      }
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 48LL))(v23);
      if ( v30 )
LABEL_26:
        sub_16D7950(v30, v20, v22);
LABEL_27:
      v58.m128i_i64[0] = (__int64)&unk_49ED7C0;
      nullsub_616(&v58, v20, v22, v24, v25);
      if ( (unsigned __int8)sub_160E750(a1 + 160) )
      {
        if ( v53 )
        {
          if ( *(_BYTE *)(a1 + 648) )
          {
            v58.m128i_i64[0] = (__int64)v59;
            sub_144A9D0(v58.m128i_i64, "<deleted>", (__int64)"");
          }
          else
          {
            sub_1442FC0(&v58, *(__int64 **)(a1 + 664));
          }
          sub_160F160(a1 + 160, v27, 1, 6, v58.m128i_i64[0], v58.m128i_i64[1]);
          if ( (_QWORD *)v58.m128i_i64[0] != v59 )
            j_j___libc_free_0(v58.m128i_i64[0], v59[0] + 1LL);
        }
        sub_1615E90(a1 + 160, v27);
      }
      if ( !*(_BYTE *)(a1 + 648) )
      {
        v32 = sub_1612E30(v27);
        v33 = v32;
        if ( v32 )
        {
          sub_16D7910(v32);
          sub_1403F30(&v58, v27, *(_QWORD *)(a1 + 168));
          sub_1444BB0(*(__int64 **)(a1 + 664));
          if ( v58.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v58.m128i_i64[0] + 48LL))(v58.m128i_i64[0]);
          sub_16D7950(v33, v27, v34);
        }
        else
        {
          sub_1403F30(&v58, v27, *(_QWORD *)(a1 + 168));
          sub_1444BB0(*(__int64 **)(a1 + 664));
          if ( v58.m128i_i64[0] )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v58.m128i_i64[0] + 48LL))(v58.m128i_i64[0]);
        }
        nullsub_568(a1 + 160, v27);
      }
      sub_16145F0(a1 + 160, v27);
      sub_16176C0(a1 + 160, v27);
      if ( (unsigned __int8)sub_160E750(a1 + 160) && !*(_BYTE *)(a1 + 648) )
      {
        sub_1442FC0(&v58, *(__int64 **)(a1 + 664));
      }
      else
      {
        v58.m128i_i64[0] = (__int64)v59;
        sub_144A9D0(v58.m128i_i64, "<deleted>", (__int64)"");
      }
      sub_1615450(a1 + 160, v27, v58.m128i_i64[0], v58.m128i_i64[1], 6);
      if ( (_QWORD *)v58.m128i_i64[0] != v59 )
        j_j___libc_free_0(v58.m128i_i64[0], v59[0] + 1LL);
      v26 = *(_DWORD *)(a1 + 192);
      if ( *(_BYTE *)(a1 + 648) )
        break;
      if ( ++v56 >= v26 )
        goto LABEL_60;
    }
    v35 = 0;
    if ( v26 )
    {
      do
      {
        v36 = v35++;
        sub_16151B0(a1 + 160, *(_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v36), "<deleted>", 9, 6);
      }
      while ( v35 < *(_DWORD *)(a1 + 192) );
    }
LABEL_60:
    v18 = *(_QWORD *)(a1 + 624);
LABEL_61:
    v37 = *(_QWORD *)(a1 + 616);
    if ( v37 == v18 )
    {
      j_j___libc_free_0(v18, 512);
      v44 = (__int64 *)(*(_QWORD *)(a1 + 640) - 8LL);
      *(_QWORD *)(a1 + 640) = v44;
      v45 = *v44;
      v46 = *v44 + 512;
      *(_QWORD *)(a1 + 624) = v45;
      *(_QWORD *)(a1 + 632) = v46;
      *(_QWORD *)(a1 + 616) = v45 + 504;
    }
    else
    {
      *(_QWORD *)(a1 + 616) = v37 - 8;
    }
    if ( *(_BYTE *)(a1 + 649) )
    {
      v43 = *(_QWORD **)(a1 + 616);
      if ( v43 == (_QWORD *)(*(_QWORD *)(a1 + 632) - 8LL) )
      {
        v47 = *(_QWORD *)(a1 + 640);
        if ( ((__int64)(*(_QWORD *)(a1 + 600) - *(_QWORD *)(a1 + 584)) >> 3)
           + ((((v47 - *(_QWORD *)(a1 + 608)) >> 3) - 1) << 6)
           + (((__int64)v43 - *(_QWORD *)(a1 + 624)) >> 3) == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"cannot create std::deque larger than max_size()");
        if ( (unsigned __int64)(*(_QWORD *)(a1 + 576) - ((v47 - *(_QWORD *)(a1 + 568)) >> 3)) <= 1 )
        {
          sub_144B360((__int64 *)(a1 + 568), 1u, 0);
          v47 = *(_QWORD *)(a1 + 640);
        }
        *(_QWORD *)(v47 + 8) = sub_22077B0(512);
        v48 = *(_QWORD **)(a1 + 616);
        if ( v48 )
          *v48 = *(_QWORD *)(a1 + 664);
        v49 = (__int64 *)(*(_QWORD *)(a1 + 640) + 8LL);
        *(_QWORD *)(a1 + 640) = v49;
        v50 = *v49;
        v51 = *v49 + 512;
        *(_QWORD *)(a1 + 624) = v50;
        *(_QWORD *)(a1 + 632) = v51;
        *(_QWORD *)(a1 + 616) = v50;
      }
      else
      {
        if ( v43 )
        {
          *v43 = *(_QWORD *)(a1 + 664);
          v43 = *(_QWORD **)(a1 + 616);
        }
        *(_QWORD *)(a1 + 616) = v43 + 1;
      }
    }
    v38 = *(_QWORD **)(*(_QWORD *)(a1 + 656) + 32LL);
    if ( v38 )
      sub_14439E0(v38);
    v17 = *(_QWORD *)(a1 + 616);
  }
  while ( v17 != *(_QWORD *)(a1 + 584) );
LABEL_67:
  if ( *(_DWORD *)(a1 + 192) )
  {
    v39 = v53;
    v40 = 0;
    do
    {
      while ( 1 )
      {
        v41 = *(__int64 (**)())(**(_QWORD **)(*(_QWORD *)(a1 + 184) + 8LL * v40) + 160LL);
        if ( v41 != sub_144A6F0 )
          break;
        if ( ++v40 >= *(_DWORD *)(a1 + 192) )
          return (unsigned __int8)v39;
      }
      ++v40;
      v39 |= v41();
    }
    while ( v40 < *(_DWORD *)(a1 + 192) );
    return (unsigned __int8)v39;
  }
  return v53;
}
