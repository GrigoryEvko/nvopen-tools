// Function: sub_269D460
// Address: 0x269d460
//
__int64 __fastcall sub_269D460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __m128i v6; // xmm1
  _QWORD *v7; // rax
  __int64 v8; // r8
  __int64 (__fastcall *v9)(__int64); // rax
  _BYTE *v10; // rdi
  __int64 (*v11)(void); // rax
  char v12; // al
  __int64 v14; // rax
  __int64 v15; // rcx
  int v16; // eax
  int v17; // edx
  int v18; // edi
  unsigned int v19; // eax
  void *v20; // rsi
  __int64 v21; // rax
  char v22; // al
  __int64 v23; // rax
  __int64 v24; // r13
  __m128i v25; // xmm2
  __m128i *v26; // rax
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // r8
  __int64 v32; // rax
  __int64 v33; // r14
  int v34; // ebx
  __int64 (__fastcall *v35)(__int64); // rax
  _BYTE *v36; // rdi
  __int64 (*v37)(void); // rax
  char v38; // al
  __int64 v39; // rdx
  char v40; // al
  __int64 (__fastcall *v41)(__int64); // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 (__fastcall *v44)(__int64); // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // [rsp+0h] [rbp-80h]
  __int64 v49; // [rsp+8h] [rbp-78h]
  char v50; // [rsp+8h] [rbp-78h]
  __int64 v51; // [rsp+8h] [rbp-78h]
  __int64 v52; // [rsp+8h] [rbp-78h]
  __int64 v53; // [rsp+8h] [rbp-78h]
  __int64 v55; // [rsp+10h] [rbp-70h]
  __int64 v56; // [rsp+10h] [rbp-70h]
  __int64 v58; // [rsp+18h] [rbp-68h]
  __int64 v59; // [rsp+18h] [rbp-68h]
  __int64 v60; // [rsp+18h] [rbp-68h]
  __m128i v61; // [rsp+20h] [rbp-60h] BYREF
  void *v62; // [rsp+30h] [rbp-50h] BYREF
  __m128i v63; // [rsp+38h] [rbp-48h]

  v61.m128i_i64[0] = a2;
  v61.m128i_i64[1] = a3;
  if ( !(unsigned __int8)sub_250E300(a1, &v61) )
    v61.m128i_i64[1] = 0;
  v6 = _mm_load_si128(&v61);
  v62 = &unk_438FC86;
  v63 = v6;
  v7 = sub_25134D0(a1 + 136, (__int64 *)&v62);
  if ( v7 )
  {
    v8 = v7[3];
    if ( v8 )
    {
      if ( a5 != 2 && a4 )
      {
        v9 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v8 + 40LL);
        if ( v9 == sub_2505F20 )
        {
          v10 = (_BYTE *)(v8 + 88);
        }
        else
        {
          v56 = v8;
          v43 = v9(v8);
          v8 = v56;
          v10 = (_BYTE *)v43;
        }
        v11 = *(__int64 (**)(void))(*(_QWORD *)v10 + 16LL);
        if ( (char *)v11 == (char *)sub_2505E30 )
        {
          v12 = v10[9];
        }
        else
        {
          v55 = v8;
          v12 = v11();
          v8 = v55;
        }
        if ( v12 )
        {
          v39 = a4;
          v58 = v8;
          sub_250ED80(a1, v8, v39, 1);
          return v58;
        }
      }
      return v8;
    }
  }
  v14 = *(_QWORD *)(a1 + 4376);
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 8);
    v16 = *(_DWORD *)(v14 + 24);
    if ( !v16 )
      return 0;
    v17 = v16 - 1;
    v18 = 1;
    v19 = (v16 - 1) & (((unsigned int)&unk_438FC86 >> 9) ^ ((unsigned int)&unk_438FC86 >> 4));
    v20 = *(void **)(v15 + 8LL * v19);
    if ( v20 != &unk_438FC86 )
    {
      while ( v20 != (void *)-4096LL )
      {
        v19 = v17 & (v18 + v19);
        v20 = *(void **)(v15 + 8LL * v19);
        if ( v20 == &unk_438FC86 )
          goto LABEL_17;
        ++v18;
      }
      return 0;
    }
  }
LABEL_17:
  v21 = sub_25096F0(&v61);
  if ( v21 )
  {
    v49 = v21;
    if ( (unsigned __int8)sub_B2D610(v21, 20) || (unsigned __int8)sub_B2D610(v49, 48) )
      return 0;
  }
  if ( *(_DWORD *)(a1 + 3556) > dword_4FEEF68[0] )
    return 0;
  v50 = sub_2673B80(a1, v61.m128i_i64);
  v22 = sub_2509800(&v61);
  if ( v22 != 4 )
  {
    if ( v22 > 4 )
    {
      if ( (unsigned __int8)(v22 - 5) <= 2u )
LABEL_37:
        BUG();
    }
    else if ( (unsigned __int8)v22 <= 3u )
    {
      goto LABEL_37;
    }
LABEL_56:
    v62 = &unk_438FC86;
    BUG();
  }
  v23 = sub_A777F0(0x100u, *(__int64 **)(a1 + 128));
  v24 = v23;
  if ( !v23 )
    goto LABEL_56;
  v25 = _mm_loadu_si128(&v61);
  v26 = (__m128i *)(v23 + 56);
  v26[-3].m128i_i64[0] = 0;
  v26[1] = v25;
  v26[-3].m128i_i64[1] = 0;
  v26[-2].m128i_i64[0] = 0;
  v26[-2].m128i_i32[2] = 0;
  *(_QWORD *)(v24 + 40) = v26;
  *(_QWORD *)(v24 + 48) = 0x200000000LL;
  *(_QWORD *)v24 = off_4A20228;
  *(_QWORD *)(v24 + 88) = &unk_4A202B8;
  *(_QWORD *)(v24 + 136) = v24 + 152;
  *(_QWORD *)(v24 + 144) = 0x400000000LL;
  *(_WORD *)(v24 + 96) = 256;
  *(_QWORD *)(v24 + 104) = 0;
  *(_QWORD *)(v24 + 112) = 0;
  *(_QWORD *)(v24 + 120) = 0;
  *(_DWORD *)(v24 + 128) = 0;
  *(_QWORD *)(v24 + 184) = 0;
  *(_QWORD *)(v24 + 192) = v24 + 216;
  *(_QWORD *)(v24 + 200) = 4;
  *(_DWORD *)(v24 + 208) = 0;
  *(_BYTE *)(v24 + 212) = 1;
  *(_DWORD *)(v24 + 248) = 0;
  v62 = &unk_438FC86;
  v63 = _mm_loadu_si128((const __m128i *)(v24 + 72));
  v27 = sub_2519B70(a1 + 136, (__int64)&v62);
  v31 = v24;
  *v27 = v24;
  if ( *(_DWORD *)(a1 + 3552) <= 1u )
  {
    v62 = (void *)(v24 & 0xFFFFFFFFFFFFFFFBLL);
    sub_269CF50(a1 + 224, (unsigned __int64 *)&v62, v28, v29, v24, v30);
    v31 = v24;
    if ( !*(_DWORD *)(a1 + 3552) )
    {
      v40 = sub_250E880(a1, v24);
      v31 = v24;
      if ( !v40 )
      {
        v41 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 40LL);
        if ( v41 == sub_2505F20 )
        {
          v42 = v24 + 88;
        }
        else
        {
          v47 = v41(v24);
          v31 = v24;
          v42 = v47;
        }
        goto LABEL_46;
      }
    }
  }
  v48 = v31;
  v62 = (void *)v24;
  v32 = sub_C99770("initialize", 10, (void (__fastcall *)(__m128i **, __int64))sub_26756C0, (__int64)&v62);
  ++*(_DWORD *)(a1 + 3556);
  v33 = v32;
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v24 + 24LL))(v24, a1);
  v31 = v48;
  --*(_DWORD *)(a1 + 3556);
  if ( v33 )
  {
    sub_C9AF60(v33);
    v31 = v48;
  }
  if ( !v50 )
  {
    v44 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 40LL);
    if ( v44 == sub_2505F20 )
    {
      v42 = v24 + 88;
    }
    else
    {
      v60 = v31;
      v45 = v44(v24);
      v31 = v60;
      v42 = v45;
    }
LABEL_46:
    v59 = v31;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v42 + 40LL))(v42);
    return v59;
  }
  v34 = *(_DWORD *)(a1 + 3552);
  v51 = v31;
  *(_DWORD *)(a1 + 3552) = 1;
  sub_251C580(a1, v24);
  v8 = v51;
  *(_DWORD *)(a1 + 3552) = v34;
  if ( a4 )
  {
    v35 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v24 + 40LL);
    if ( v35 == sub_2505F20 )
    {
      v36 = (_BYTE *)(v24 + 88);
    }
    else
    {
      v46 = v35(v24);
      v8 = v51;
      v36 = (_BYTE *)v46;
    }
    v37 = *(__int64 (**)(void))(*(_QWORD *)v36 + 16LL);
    if ( (char *)v37 == (char *)sub_2505E30 )
    {
      v38 = v36[9];
    }
    else
    {
      v53 = v8;
      v38 = v37();
      v8 = v53;
    }
    if ( v38 )
    {
      v52 = v8;
      sub_250ED80(a1, v24, a4, a5);
      return v52;
    }
  }
  return v8;
}
