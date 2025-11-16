// Function: sub_3793DA0
// Address: 0x3793da0
//
__m128i *__fastcall sub_3793DA0(__int64 a1, unsigned __int64 a2, __m128i a3)
{
  unsigned int v3; // ebx
  unsigned __int16 v6; // r15
  __int64 v7; // rcx
  char v8; // al
  __int64 v9; // rdx
  __int64 v10; // r8
  __int64 v11; // rbx
  unsigned __int64 v12; // r13
  __m128i *v13; // r12
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r9
  __int64 v18; // r11
  __int64 (__fastcall *v19)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int64 v20; // rax
  char v21; // cl
  unsigned __int64 v22; // rsi
  __int64 *v23; // rdi
  unsigned int v24; // eax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rcx
  char v28; // r8
  __int64 v29; // rdi
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rdi
  __int128 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // rdi
  __int64 (*v37)(void); // rdx
  unsigned __int16 v38; // ax
  __int128 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rsi
  _QWORD *v43; // r15
  __int64 v44; // r8
  __int64 v45; // r9
  unsigned __int8 *v46; // r15
  unsigned int v47; // edx
  __int64 v48; // r8
  __int64 v49; // rdx
  __int128 v50; // [rsp-10h] [rbp-1F0h]
  __int64 v51; // [rsp+0h] [rbp-1E0h]
  __int64 v52; // [rsp+8h] [rbp-1D8h]
  __int64 v53; // [rsp+18h] [rbp-1C8h]
  char v54; // [rsp+20h] [rbp-1C0h]
  __int64 v55; // [rsp+20h] [rbp-1C0h]
  __int128 v56; // [rsp+20h] [rbp-1C0h]
  unsigned __int8 *v57; // [rsp+20h] [rbp-1C0h]
  unsigned int v58; // [rsp+38h] [rbp-1A8h]
  __int64 v59; // [rsp+50h] [rbp-190h] BYREF
  __int64 v60; // [rsp+58h] [rbp-188h]
  unsigned int v61; // [rsp+60h] [rbp-180h] BYREF
  __int64 v62; // [rsp+68h] [rbp-178h]
  __int64 v63; // [rsp+70h] [rbp-170h] BYREF
  int v64; // [rsp+78h] [rbp-168h]
  __int64 v65; // [rsp+80h] [rbp-160h]
  __int64 v66; // [rsp+88h] [rbp-158h]
  __int64 v67; // [rsp+90h] [rbp-150h]
  __int64 v68; // [rsp+98h] [rbp-148h]
  __int64 *v69; // [rsp+A0h] [rbp-140h] BYREF
  __int64 v70; // [rsp+A8h] [rbp-138h]
  __int64 v71; // [rsp+B0h] [rbp-130h] BYREF
  unsigned int v72; // [rsp+B8h] [rbp-128h]

  v6 = *(_WORD *)(a2 + 96);
  v7 = *(_QWORD *)(a2 + 104);
  v8 = *(_BYTE *)(a2 + 33) >> 2;
  LOWORD(v69) = v6;
  v70 = v7;
  v54 = v8 & 3;
  if ( v6 )
  {
    if ( v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
      BUG();
    v15 = *(_QWORD *)&byte_444C4A0[16 * v6 - 16];
    if ( !v15 )
      goto LABEL_3;
  }
  else
  {
    v53 = v7;
    v65 = sub_3007260((__int64)&v69);
    v66 = v9;
    if ( !v65 )
    {
LABEL_3:
      v10 = *(_QWORD *)(a1 + 8);
LABEL_4:
      sub_3460140((__int64)&v69, a3, *(_QWORD *)a1, a2, v10);
      v11 = v72;
      v12 = v71;
      sub_3760E70(a1, a2, 0, (unsigned __int64)v69, (unsigned int)v70);
      v13 = 0;
      sub_3760E70(a1, a2, 1, v12, v11);
      return v13;
    }
    v15 = sub_3007260((__int64)&v69);
    v7 = v53;
    v67 = v15;
    v68 = v16;
  }
  v17 = *(_QWORD *)a1;
  v10 = *(_QWORD *)(a1 + 8);
  if ( (v15 & 7) != 0 )
    goto LABEL_4;
  LOWORD(v59) = v6;
  v18 = *(_QWORD *)(v10 + 64);
  v60 = v7;
  v19 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v17 + 592LL);
  if ( v19 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v69, v17, v18, v59, v60);
    LOWORD(v20) = v70;
    LOWORD(v61) = v70;
    v62 = v71;
  }
  else
  {
    LODWORD(v20) = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v19)(v17, v18, (unsigned int)v59);
    v61 = v20;
    v62 = v49;
  }
  if ( (_WORD)v20 )
  {
    v21 = (unsigned __int16)(v20 - 176) <= 0x34u;
    LODWORD(v22) = word_4456340[(unsigned __int16)v20 - 1];
    LOBYTE(v20) = v21;
  }
  else
  {
    v22 = sub_3007240((__int64)&v61);
    v20 = HIDWORD(v22);
    v21 = BYTE4(v22);
  }
  v23 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  LODWORD(v69) = v22;
  BYTE4(v69) = v20;
  if ( v21 )
  {
    LOWORD(v24) = sub_2D43AD0(2, v22);
    v27 = 0;
    if ( (_WORD)v24 )
      goto LABEL_14;
LABEL_27:
    v24 = sub_3009450(v23, 2, 0, (__int64)v69, v25, v26);
    v28 = v54;
    v27 = v41;
    v3 = v24;
    if ( !v54 )
      goto LABEL_15;
LABEL_28:
    v69 = &v71;
    v70 = 0x1000000000LL;
    v57 = sub_378FB20((__int64 **)a1, (__int64)&v69, a2, v28);
    goto LABEL_29;
  }
  LOWORD(v24) = sub_2D43050(2, v22);
  v27 = 0;
  if ( !(_WORD)v24 )
    goto LABEL_27;
LABEL_14:
  v28 = v54;
  LOWORD(v3) = v24;
  if ( v54 )
    goto LABEL_28;
LABEL_15:
  v29 = *(_QWORD *)a1;
  v30 = 1;
  if ( (_WORD)v61 == 1
    || (_WORD)v61 && (v30 = (unsigned __int16)v61, *(_QWORD *)(v29 + 8LL * (unsigned __int16)v61 + 112)) )
  {
    if ( (*(_BYTE *)(v29 + 500 * v30 + 6882) & 0xFB) == 0
      && (_WORD)v24
      && *(_QWORD *)(v29 + 8LL * (unsigned __int16)v24 + 112) )
    {
      v31 = *(_QWORD *)(a2 + 80);
      v69 = (__int64 *)v31;
      if ( v31 )
      {
        v55 = v27;
        sub_B96E90((__int64)&v69, v31, 1);
        v27 = v55;
      }
      v32 = *(_QWORD *)(a1 + 8);
      LODWORD(v70) = *(_DWORD *)(a2 + 72);
      *(_QWORD *)&v33 = sub_34015B0(v32, (__int64)&v69, v3, v27, 0, 0, a3);
      v34 = *(_QWORD *)(a1 + 8);
      v56 = v33;
      v35 = sub_3281590((__int64)&v59);
      v36 = *(_QWORD *)a1;
      v63 = v35;
      v37 = *(__int64 (**)(void))(*(_QWORD *)v36 + 80LL);
      v38 = 7;
      if ( v37 != sub_2FE2E20 )
        v38 = v37();
      *(_QWORD *)&v39 = sub_3401C20(v34, (__int64)&v69, v38, 0, v63, a3);
      v40 = a1;
      v13 = sub_33E9660(
              *(__int64 **)(a1 + 8),
              (*(_WORD *)(a2 + 32) >> 7) & 7,
              0,
              v61,
              v62,
              (__int64)&v69,
              *(_OWORD *)*(_QWORD *)(a2 + 40),
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
              v56,
              v39,
              *(unsigned __int16 *)(a2 + 96),
              *(_QWORD *)(a2 + 104),
              *(const __m128i **)(a2 + 112),
              0);
      sub_3760E70(v40, a2, 1, (unsigned __int64)v13, 1);
      if ( v69 )
        sub_B91220((__int64)&v69, (__int64)v69);
      return v13;
    }
  }
  v69 = &v71;
  v70 = 0x1000000000LL;
  v57 = sub_37929A0((__int64 *)a1, (__int64)&v69, a2, a3);
LABEL_29:
  if ( !v57 )
    sub_C64ED0("Unable to widen vector load", 1u);
  if ( (unsigned int)v70 == 1 )
  {
    v46 = (unsigned __int8 *)*v69;
    v48 = *((unsigned int *)v69 + 2);
  }
  else
  {
    v42 = *(_QWORD *)(a2 + 80);
    v43 = *(_QWORD **)(a1 + 8);
    v44 = (__int64)v69;
    v45 = (unsigned int)v70;
    v63 = v42;
    if ( v42 )
    {
      v51 = (__int64)v69;
      v52 = (unsigned int)v70;
      sub_B96E90((__int64)&v63, v42, 1);
      v44 = v51;
      v45 = v52;
    }
    *((_QWORD *)&v50 + 1) = v45;
    *(_QWORD *)&v50 = v44;
    v64 = *(_DWORD *)(a2 + 72);
    v46 = sub_33FC220(v43, 2, (__int64)&v63, 1, 0, v45, v50);
    v58 = v47;
    v48 = v47;
    if ( v63 )
    {
      sub_B91220((__int64)&v63, v63);
      v48 = v58;
    }
  }
  sub_3760E70(a1, a2, 1, (unsigned __int64)v46, v48);
  v13 = (__m128i *)v57;
  if ( v69 != &v71 )
    _libc_free((unsigned __int64)v69);
  return v13;
}
