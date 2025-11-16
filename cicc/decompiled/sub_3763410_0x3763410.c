// Function: sub_3763410
// Address: 0x3763410
//
unsigned __int8 *__fastcall sub_3763410(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        int a7)
{
  __int64 v9; // rsi
  __int16 *v10; // rax
  __int16 v11; // dx
  __int64 v12; // r8
  unsigned int *v13; // rax
  __int64 v14; // rax
  unsigned __int16 v15; // dx
  __int64 v16; // rax
  __int64 v17; // rdi
  unsigned __int8 *v18; // rax
  unsigned __int16 v19; // bx
  unsigned __int8 *v20; // r12
  __int64 v21; // rdx
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rbx
  unsigned __int16 v29; // dx
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int128 v33; // rax
  _QWORD *v34; // r15
  __int64 v35; // r9
  __int128 v36; // rax
  __int64 v37; // r9
  unsigned __int8 *v38; // r12
  bool v40; // al
  __int64 v41; // rcx
  __int64 v42; // r8
  unsigned __int16 v43; // ax
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int128 v49; // [rsp-30h] [rbp-E0h]
  __int128 v50; // [rsp+10h] [rbp-A0h]
  __int64 v51; // [rsp+20h] [rbp-90h] BYREF
  int v52; // [rsp+28h] [rbp-88h]
  unsigned int v53; // [rsp+30h] [rbp-80h] BYREF
  __int64 v54; // [rsp+38h] [rbp-78h]
  unsigned __int16 v55; // [rsp+40h] [rbp-70h] BYREF
  __int64 v56; // [rsp+48h] [rbp-68h]
  unsigned __int16 v57; // [rsp+50h] [rbp-60h] BYREF
  __int64 v58; // [rsp+58h] [rbp-58h]
  __int64 v59; // [rsp+60h] [rbp-50h]
  __int64 v60; // [rsp+68h] [rbp-48h]
  __int64 v61; // [rsp+70h] [rbp-40h] BYREF
  __int64 v62; // [rsp+78h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v51 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v51, v9, 1);
  v52 = *(_DWORD *)(a2 + 72);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = *(unsigned int **)(a2 + 40);
  v54 = v12;
  LOWORD(v53) = v11;
  v14 = *(_QWORD *)(*(_QWORD *)v13 + 48LL) + 16LL * v13[2];
  v15 = *(_WORD *)v14;
  v16 = *(_QWORD *)(v14 + 8);
  v17 = *a1;
  v55 = v15;
  v56 = v16;
  v18 = sub_33FAF80(v17, 223, (__int64)&v51, v53, v12, a7, a3);
  v19 = v53;
  v20 = v18;
  v22 = v21;
  if ( (_WORD)v53 )
  {
    if ( (unsigned __int16)(v53 - 17) <= 0xD3u )
    {
      v62 = 0;
      v19 = word_4456580[(unsigned __int16)v53 - 1];
      LOWORD(v61) = v19;
      if ( !v19 )
        goto LABEL_7;
      goto LABEL_22;
    }
    goto LABEL_5;
  }
  if ( !sub_30070B0((__int64)&v53) )
  {
LABEL_5:
    v23 = v54;
    goto LABEL_6;
  }
  v19 = sub_3009970((__int64)&v53, 223, v46, v47, v48);
LABEL_6:
  LOWORD(v61) = v19;
  v62 = v23;
  if ( !v19 )
  {
LABEL_7:
    v24 = sub_3007260((__int64)&v61);
    v28 = v25;
    v26 = v24;
    v27 = v28;
    v59 = v26;
    LODWORD(v28) = v26;
    v60 = v27;
    goto LABEL_8;
  }
LABEL_22:
  if ( v19 == 1 || (unsigned __int16)(v19 - 504) <= 7u )
LABEL_28:
    BUG();
  v28 = *(_QWORD *)&byte_444C4A0[16 * v19 - 16];
LABEL_8:
  v29 = v55;
  if ( !v55 )
  {
    v40 = sub_30070B0((__int64)&v55);
    v29 = 0;
    if ( v40 )
    {
      v43 = sub_3009970((__int64)&v55, 223, 0, v41, v42);
      v45 = v44;
      v29 = v43;
      v57 = v43;
      v58 = v45;
      if ( !v43 )
        goto LABEL_12;
      goto LABEL_18;
    }
    goto LABEL_10;
  }
  if ( (unsigned __int16)(v55 - 17) > 0xD3u )
  {
LABEL_10:
    v30 = v56;
    goto LABEL_11;
  }
  v29 = word_4456580[v55 - 1];
  v30 = 0;
LABEL_11:
  v57 = v29;
  v58 = v30;
  if ( !v29 )
  {
LABEL_12:
    v31 = sub_3007260((__int64)&v57);
    v61 = v31;
    v62 = v32;
    goto LABEL_13;
  }
LABEL_18:
  if ( v29 == 1 || (unsigned __int16)(v29 - 504) <= 7u )
    goto LABEL_28;
  v31 = *(_QWORD *)&byte_444C4A0[16 * v29 - 16];
LABEL_13:
  *(_QWORD *)&v33 = sub_3400BD0(*a1, (unsigned int)(v28 - v31), (__int64)&v51, v53, v54, 0, a3, 0);
  v34 = (_QWORD *)*a1;
  *((_QWORD *)&v49 + 1) = v22;
  *(_QWORD *)&v49 = v20;
  v50 = v33;
  *(_QWORD *)&v36 = sub_3406EB0(v34, 0xBEu, (__int64)&v51, v53, v54, v35, v49, v33);
  v38 = sub_3406EB0(v34, 0xBFu, (__int64)&v51, v53, v54, v37, v36, v50);
  if ( v51 )
    sub_B91220((__int64)&v51, v51);
  return v38;
}
