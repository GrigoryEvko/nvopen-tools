// Function: sub_26A2E60
// Address: 0x26a2e60
//
__int64 __fastcall sub_26A2E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  int v8; // ecx
  int v9; // ecx
  int v10; // ebx
  unsigned int i; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  int v17; // edx
  int v18; // r8d
  unsigned int v19; // eax
  void *v20; // rdi
  __int64 v21; // rax
  __int64 v22; // rbx
  __int64 v23; // rdx
  int v24; // eax
  unsigned __int64 v25; // rax
  __int64 v26; // rcx
  __int64 *v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rbx
  __m128i v30; // xmm4
  __m128i *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rax
  int v37; // r15d
  __int64 (__fastcall *v39)(__int64); // rax
  _BYTE *v40; // rdi
  __int64 (*v41)(void); // rax
  char v42; // al
  __int64 *v43; // rsi
  __int64 v44; // rax
  __m128i v45; // xmm5
  __m128i *v46; // rax
  _QWORD *v47; // rax
  __int64 (__fastcall *v48)(__int64); // rax
  __int64 v49; // rdi
  __int64 v50; // rax
  __m128i v51; // xmm3
  __m128i *v52; // rax
  __int64 (__fastcall **v53)(); // rax
  __int64 v54; // rax
  __m128i v55; // xmm2
  __m128i *v56; // rax
  __int64 v57; // [rsp+0h] [rbp-70h]
  char v58; // [rsp+8h] [rbp-68h]
  __m128i v59; // [rsp+10h] [rbp-60h] BYREF
  void *v60; // [rsp+20h] [rbp-50h] BYREF
  __m128i v61; // [rsp+28h] [rbp-48h]

  v59.m128i_i64[0] = a2;
  v59.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v59) )
    v59.m128i_i64[1] = 0;
  v8 = *(_DWORD *)(a1 + 160);
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = 1;
    for ( i = v9
            & (((0xBF58476D1CE4E5B9LL
               * (((unsigned __int64)(((unsigned int)&unk_438FC89 >> 9) ^ ((unsigned int)&unk_438FC89 >> 4)) << 32)
                | ((unsigned __int32)v59.m128i_i32[2] >> 9)
                ^ ((unsigned __int32)v59.m128i_i32[2] >> 4)
                ^ (16 * (((unsigned __int32)v59.m128i_i32[0] >> 9) ^ ((unsigned __int32)v59.m128i_i32[0] >> 4))))) >> 31)
             ^ (484763065
              * (((unsigned __int32)v59.m128i_i32[2] >> 9)
               ^ ((unsigned __int32)v59.m128i_i32[2] >> 4)
               ^ (16 * (((unsigned __int32)v59.m128i_i32[0] >> 9) ^ ((unsigned __int32)v59.m128i_i32[0] >> 4))))));
          ;
          i = v9 & v13 )
    {
      v12 = *(_QWORD *)(a1 + 144) + 32LL * i;
      if ( *(_UNKNOWN **)v12 == &unk_438FC89 && *(_OWORD *)&v59 == *(_OWORD *)(v12 + 8) )
        break;
      if ( *(_QWORD *)v12 == -4096 && *(_QWORD *)(v12 + 8) == qword_4FEE4D0 && *(_QWORD *)(v12 + 16) == qword_4FEE4D8 )
        goto LABEL_10;
      v13 = v10 + i;
      ++v10;
    }
    v29 = *(_QWORD *)(v12 + 24);
    if ( v29 )
    {
      if ( a5 == 2 )
        return v29;
      goto LABEL_38;
    }
  }
LABEL_10:
  v14 = *(_QWORD *)(a1 + 4376);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 8);
    v16 = *(_DWORD *)(v14 + 24);
    if ( !v16 )
      return 0;
    v17 = v16 - 1;
    v18 = 1;
    v19 = (v16 - 1) & (((unsigned int)&unk_438FC89 >> 9) ^ ((unsigned int)&unk_438FC89 >> 4));
    v20 = *(void **)(v15 + 8LL * v19);
    if ( v20 != &unk_438FC89 )
    {
      while ( v20 != (void *)-4096LL )
      {
        v19 = v17 & (v18 + v19);
        v20 = *(void **)(v15 + 8LL * v19);
        if ( v20 == &unk_438FC89 )
          goto LABEL_13;
        ++v18;
      }
      return 0;
    }
  }
LABEL_13:
  v21 = sub_25096F0(&v59);
  v22 = v21;
  if ( v21 && ((unsigned __int8)sub_B2D610(v21, 20) || (unsigned __int8)sub_B2D610(v22, 48))
    || *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
  {
    return 0;
  }
  v58 = sub_2673B80(a1, v59.m128i_i64);
  v23 = v59.m128i_i8[0] & 3;
  if ( v23 == 2 )
    goto LABEL_70;
  if ( v23 == 3 )
    goto LABEL_70;
  if ( (v59.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL) == 0 )
    goto LABEL_70;
  v24 = *(unsigned __int8 *)(v59.m128i_i64[0] & 0xFFFFFFFFFFFFFFFCLL);
  if ( (_BYTE)v24 == 22 )
    goto LABEL_70;
  if ( (_BYTE)v24 )
  {
    if ( (unsigned __int8)v24 > 0x1Cu )
    {
      v25 = (unsigned int)(v24 - 34);
      if ( (unsigned __int8)v25 <= 0x33u )
      {
        v26 = 0x8000000000041LL;
        if ( _bittest64(&v26, v25) )
        {
          v27 = *(__int64 **)(a1 + 128);
          if ( (_BYTE)v23 != 1 )
          {
            v28 = sub_A777F0(0x80u, v27);
            v29 = v28;
            if ( v28 )
            {
              v30 = _mm_loadu_si128(&v59);
              v31 = (__m128i *)(v28 + 56);
              v31[-3].m128i_i64[0] = 0;
              v31[-3].m128i_i64[1] = 0;
              v31[1] = v30;
              v31[-2].m128i_i64[0] = 0;
              v31[-2].m128i_i32[2] = 0;
              *(_QWORD *)(v29 + 40) = v31;
              *(_QWORD *)(v29 + 48) = 0x200000000LL;
              *(_QWORD *)v29 = off_4A1FF38;
              *(_WORD *)(v29 + 96) = 256;
              *(_DWORD *)(v29 + 100) = 0;
              *(_QWORD *)(v29 + 88) = &unk_4A1FFC8;
              *(_BYTE *)(v29 + 120) = 0;
              goto LABEL_28;
            }
LABEL_71:
            v60 = &unk_438FC89;
            BUG();
          }
          v50 = sub_A777F0(0xB8u, v27);
          v29 = v50;
          if ( !v50 )
            goto LABEL_71;
          v51 = _mm_loadu_si128(&v59);
          *(_QWORD *)(v50 + 8) = 0;
          v52 = (__m128i *)(v50 + 56);
          v52[-3].m128i_i64[1] = 0;
          v52[1] = v51;
          v52[-2].m128i_i64[0] = 0;
          v52[-2].m128i_i32[2] = 0;
          *(_WORD *)(v29 + 96) = 256;
          *(_DWORD *)(v29 + 100) = 0;
          *(_QWORD *)(v29 + 40) = v52;
          *(_QWORD *)(v29 + 48) = 0x200000000LL;
          v53 = off_4A20028;
LABEL_62:
          *(_QWORD *)v29 = v53;
          *(_QWORD *)(v29 + 88) = v53 + 18;
          *(_OWORD *)(v29 + 104) = 0;
          *(_OWORD *)(v29 + 120) = 0;
          *(_OWORD *)(v29 + 136) = 0;
          *(_OWORD *)(v29 + 152) = 0;
          *(_OWORD *)(v29 + 168) = 0;
          goto LABEL_28;
        }
      }
    }
LABEL_70:
    BUG();
  }
  v43 = *(__int64 **)(a1 + 128);
  if ( (_BYTE)v23 == 1 )
  {
    v54 = sub_A777F0(0xB8u, v43);
    v29 = v54;
    if ( !v54 )
      goto LABEL_71;
    v55 = _mm_loadu_si128(&v59);
    v56 = (__m128i *)(v54 + 56);
    v56[-3].m128i_i64[0] = 0;
    v56[-3].m128i_i64[1] = 0;
    v56[1] = v55;
    v56[-2].m128i_i64[0] = 0;
    v56[-2].m128i_i32[2] = 0;
    *(_QWORD *)(v29 + 40) = v56;
    *(_QWORD *)(v29 + 48) = 0x200000000LL;
    v53 = off_4A1FE48;
    *(_WORD *)(v29 + 96) = 256;
    *(_DWORD *)(v29 + 100) = 0;
    goto LABEL_62;
  }
  v44 = sub_A777F0(0x108u, v43);
  v29 = v44;
  if ( !v44 )
    goto LABEL_71;
  v45 = _mm_loadu_si128(&v59);
  v46 = (__m128i *)(v44 + 56);
  v46[-3].m128i_i64[0] = 0;
  v46[-3].m128i_i64[1] = 0;
  v46[1] = v45;
  v46[-2].m128i_i64[0] = 0;
  v46[-2].m128i_i32[2] = 0;
  *(_QWORD *)(v29 + 40) = v46;
  *(_QWORD *)(v29 + 48) = 0x200000000LL;
  *(_DWORD *)(v29 + 100) = 0;
  *(_QWORD *)v29 = off_4A1FD58;
  *(_WORD *)(v29 + 96) = 256;
  *(_QWORD *)(v29 + 88) = &unk_4A1FDE8;
  v47 = (_QWORD *)(v29 + 104);
  do
  {
    *v47 = 0;
    v47 += 4;
    *((_DWORD *)v47 - 2) = 0;
    *(v47 - 3) = 0;
    *((_DWORD *)v47 - 4) = 0;
    *((_DWORD *)v47 - 3) = 0;
  }
  while ( (_QWORD *)(v29 + 264) != v47 );
LABEL_28:
  v60 = &unk_438FC89;
  v61 = _mm_loadu_si128((const __m128i *)(v29 + 72));
  *sub_2519B70(a1 + 136, (__int64)&v60) = v29;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v60 = (void *)(v29 & 0xFFFFFFFFFFFFFFFBLL);
    sub_269CF50(a1 + 224, (unsigned __int64 *)&v60, v32, v33, v34, v35);
    if ( !*(_DWORD *)(a1 + 3552) && !(unsigned __int8)sub_250E880(a1, v29) )
      goto LABEL_57;
  }
  v60 = (void *)v29;
  v36 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_2675510, (__int64)&v60);
  ++*(_DWORD *)(a1 + 3556);
  v57 = v36;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v29 + 24LL))(v29, a1);
  --*(_DWORD *)(a1 + 3556);
  if ( v57 )
    sub_C9AF60(v57);
  if ( !v58 )
  {
LABEL_57:
    v48 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 40LL);
    if ( v48 == sub_2505F20 )
      v49 = v29 + 88;
    else
      v49 = v48(v29);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v49 + 40LL))(v49);
    return v29;
  }
  v37 = *(_DWORD *)(a1 + 3552);
  *(_DWORD *)(a1 + 3552) = 1;
  sub_251C580(a1, v29);
  *(_DWORD *)(a1 + 3552) = v37;
LABEL_38:
  if ( a4 )
  {
    v39 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v29 + 40LL);
    if ( v39 == sub_2505F20 )
      v40 = (_BYTE *)(v29 + 88);
    else
      v40 = (_BYTE *)v39(v29);
    v41 = *(__int64 (**)(void))(*(_QWORD *)v40 + 16LL);
    if ( (char *)v41 == (char *)sub_2505E30 )
      v42 = v40[9];
    else
      v42 = v41();
    if ( v42 )
      sub_250ED80(a1, v29, a4, a5);
  }
  return v29;
}
