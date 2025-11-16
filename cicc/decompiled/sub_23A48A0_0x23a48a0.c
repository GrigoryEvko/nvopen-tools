// Function: sub_23A48A0
// Address: 0x23a48a0
//
unsigned __int64 *__fastcall sub_23A48A0(unsigned __int64 *a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // r9
  _QWORD *v10; // rax
  __m128i *v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // rbx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  _QWORD *v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r9
  __int64 v50; // rax
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // r8
  __int64 v54; // r9
  _QWORD *v55; // rax
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rax
  __int64 v61; // r9
  __int64 v62; // rdx
  __int64 v63; // rcx
  __int64 v64; // r8
  __int64 v65; // r9
  __int64 v66; // r9
  _QWORD *v67; // rax
  _QWORD *v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rdx
  __int64 v71; // rcx
  __int64 v72; // r8
  __int64 v73; // r9
  _QWORD *v74; // rax
  _QWORD *v75; // rax
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // r9
  __int64 v81; // rdi
  __int64 v82; // rdi
  unsigned __int8 v83; // [rsp+8h] [rbp-D18h]
  char v85; // [rsp+1Ch] [rbp-D04h]
  __int16 v86; // [rsp+20h] [rbp-D00h]
  __int64 v88; // [rsp+3Ch] [rbp-CE4h]
  int v89; // [rsp+44h] [rbp-CDCh]
  __int64 v90; // [rsp+48h] [rbp-CD8h]
  int v91; // [rsp+50h] [rbp-CD0h]
  __int64 v92; // [rsp+54h] [rbp-CCCh]
  int v93; // [rsp+5Ch] [rbp-CC4h]
  __int64 v94; // [rsp+60h] [rbp-CC0h] BYREF
  __int64 v95; // [rsp+68h] [rbp-CB8h]
  __int64 v96; // [rsp+70h] [rbp-CB0h]
  __int64 v97[2]; // [rsp+80h] [rbp-CA0h] BYREF
  char v98; // [rsp+90h] [rbp-C90h] BYREF
  int v99; // [rsp+C0h] [rbp-C60h]
  unsigned __int64 v100[3]; // [rsp+C8h] [rbp-C58h] BYREF
  unsigned __int64 v101; // [rsp+E0h] [rbp-C40h] BYREF
  char *v102; // [rsp+E8h] [rbp-C38h]
  char *v103; // [rsp+F0h] [rbp-C30h]
  __m128i v104[48]; // [rsp+100h] [rbp-C20h] BYREF
  __m128i v105; // [rsp+400h] [rbp-920h] BYREF
  __int64 v106; // [rsp+410h] [rbp-910h]

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  a1[3] = 0;
  a1[4] = 0;
  if ( (unsigned __int8)sub_C92250() )
  {
    v105 = 0u;
    v106 = 0x1000000000LL;
    sub_2354710(a1, (__int64)&v105);
    sub_B72400(v105.m128i_i64, (__int64)&v105);
  }
  sub_291E720(&v105, 0);
  sub_23A2000(a1, v105.m128i_i8);
  v5 = sub_22077B0(0x10u);
  if ( v5 )
  {
    *(_BYTE *)(v5 + 8) = 1;
    *(_QWORD *)v5 = &unk_4A118B8;
  }
  v105.m128i_i64[0] = v5;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  if ( v105.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v105.m128i_i64[0] + 8LL))(v105.m128i_i64[0]);
  v104[1].m128i_i64[0] = 0;
  v104[0].m128i_i64[0] = 0x100010000000001LL;
  v104[0].m128i_i64[1] = 0x1000101000000LL;
  sub_29744D0(&v105, v104);
  sub_23A1F80(a1, v105.m128i_i64);
  LOBYTE(v88) = 0;
  HIDWORD(v88) = 1;
  LOBYTE(v89) = 0;
  sub_F10C20((__int64)&v105, v88, v89);
  sub_2353C90(a1, (__int64)&v105, v6, v7, v8, v9);
  sub_233BCC0((__int64)&v105);
  v10 = (_QWORD *)sub_22077B0(0x10u);
  if ( v10 )
    *v10 = &unk_4A0FBF8;
  v105.m128i_i64[0] = (__int64)v10;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  sub_23A0D70(a2, (__int64)a1, a3);
  v104[0].m128i_i64[0] = 0x100010000000001LL;
  v104[0].m128i_i64[1] = 0x1000101000000LL;
  v104[1].m128i_i64[0] = 0;
  sub_29744D0(&v105, v104);
  sub_23A1F80(a1, v105.m128i_i64);
  memset(v104, 0, 0x2F8u);
  sub_2350260(v104[6].m128i_i64, 0);
  v11 = &v104[11];
  do
  {
    v11->m128i_i64[0] = 0;
    v11 += 2;
    v11[-1].m128i_i32[2] = 0;
    v11[-2].m128i_i64[1] = 0;
    v11[-1].m128i_i32[0] = 0;
    v11[-1].m128i_i32[1] = 0;
  }
  while ( &v104[47] != v11 );
  sub_23504B0((__int64)&v105, v104);
  v12 = (_QWORD *)sub_22077B0(0x300u);
  v13 = (__int64)v12;
  if ( v12 )
  {
    *v12 = &unk_4A10BB8;
    sub_23504B0((__int64)(v12 + 1), &v105);
  }
  v97[0] = v13;
  sub_23A1F40(a1, (unsigned __int64 *)v97);
  sub_233EFE0(v97);
  sub_233B610((__int64)&v105);
  sub_233B610((__int64)v104);
  v97[0] = (__int64)&v98;
  v97[1] = 0x600000000LL;
  v99 = 0;
  memset(v100, 0, sizeof(v100));
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104[0].m128i_i64[0] = (__int64)v104[1].m128i_i64;
  v104[0].m128i_i64[1] = 0x600000000LL;
  v104[4].m128i_i32[0] = 0;
  v104[4].m128i_i64[1] = 0;
  memset(&v104[5], 0, 40);
  sub_2332320((__int64)v97, 0, (__int64)v104[1].m128i_i64, v14, v15, v16);
  v17 = (_QWORD *)sub_22077B0(0x10u);
  if ( v17 )
    *v17 = &unk_4A120F8;
  v105.m128i_i64[0] = (__int64)v17;
  sub_23A46F0(v100, (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  sub_2332320((__int64)v97, 0, v18, v19, v20, v21);
  v22 = (_QWORD *)sub_22077B0(0x10u);
  if ( v22 )
    *v22 = &unk_4A121F8;
  v105.m128i_i64[0] = (__int64)v22;
  sub_23A46F0(v100, (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  v23 = *(_QWORD *)(a2 + 16);
  v105.m128i_i16[4] = 0;
  v105.m128i_i64[0] = v23;
  sub_23A47E0((unsigned __int64 *)v97, v105.m128i_i64, v24, v25, v26, v27);
  sub_28448C0(&v94, 1, (a4 & 0xFFFFFFFD) == 1);
  sub_2332320((__int64)v97, 0, v28, v29, v30, v31);
  v86 = v94;
  v32 = sub_22077B0(0x10u);
  if ( v32 )
  {
    *(_WORD *)(v32 + 8) = v86;
    *(_QWORD *)v32 = &unk_4A124B8;
  }
  v105.m128i_i64[0] = v32;
  sub_23A46F0(v100, (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  v105.m128i_i64[0] = *(_QWORD *)(a2 + 16);
  v105.m128i_i16[4] = 1;
  sub_23A47E0((unsigned __int64 *)v97, v105.m128i_i64, v33, 1, v34, v35);
  sub_2332320((__int64)v97, 0, v36, v37, v38, v39);
  v40 = sub_22077B0(0x10u);
  if ( v40 )
  {
    *(_QWORD *)v40 = &unk_4A124F8;
    *(_WORD *)(v40 + 8) = 256;
  }
  v105.m128i_i64[0] = v40;
  sub_23A46F0(v100, (unsigned __int64 *)&v105);
  if ( v105.m128i_i64[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v105.m128i_i64[0] + 8LL))(v105.m128i_i64[0]);
  if ( (_BYTE)qword_4FDD5A8 )
  {
    sub_2332320((__int64)v97, 1, v41, v42, v43, v44);
    v82 = sub_22077B0(0x10u);
    if ( v82 )
      *(_QWORD *)v82 = &unk_4A13D38;
    v105.m128i_i64[0] = v82;
    if ( v102 == v103 )
    {
      sub_235B010(&v101, v102, &v105);
      v82 = v105.m128i_i64[0];
    }
    else
    {
      if ( v102 )
      {
        *(_QWORD *)v102 = v82;
        v102 += 8;
        goto LABEL_24;
      }
      v102 = (char *)8;
    }
    if ( v82 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v82 + 8LL))(v82);
  }
LABEL_24:
  sub_2332320((__int64)v104, 0, v41, v42, v43, v44);
  v45 = (_QWORD *)sub_22077B0(0x10u);
  if ( v45 )
    *v45 = &unk_4A12078;
  v105.m128i_i64[0] = (__int64)v45;
  sub_23A46F0(&v104[4].m128i_u64[1], (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  sub_2332320((__int64)v104, 0, v46, v47, v48, v49);
  v50 = sub_22077B0(0x10u);
  if ( v50 )
  {
    *(_BYTE *)(v50 + 8) = 1;
    *(_QWORD *)v50 = &unk_4A11F78;
  }
  v105.m128i_i64[0] = v50;
  sub_23A46F0(&v104[4].m128i_u64[1], (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  sub_23A0DE0(a2, (__int64)v104, a3);
  sub_2332320((__int64)v104, 0, v51, v52, v53, v54);
  v55 = (_QWORD *)sub_22077B0(0x10u);
  if ( v55 )
    *v55 = &unk_4A12038;
  v105.m128i_i64[0] = (__int64)v55;
  sub_23A46F0(&v104[4].m128i_u64[1], (unsigned __int64 *)&v105);
  sub_233F7D0(v105.m128i_i64);
  if ( *(_BYTE *)(a2 + 12) )
  {
    sub_2332320((__int64)v104, 1, v56, v57, v58, v59);
    v81 = sub_22077B0(0x10u);
    if ( v81 )
      *(_QWORD *)v81 = &unk_4A13D78;
    v105.m128i_i64[0] = v81;
    if ( v104[6].m128i_i64[1] == v104[7].m128i_i64[0] )
    {
      sub_235B010((unsigned __int64 *)&v104[6], (char *)v104[6].m128i_i64[1], &v105);
      v81 = v105.m128i_i64[0];
    }
    else
    {
      if ( v104[6].m128i_i64[1] )
      {
        *(_QWORD *)v104[6].m128i_i64[1] = v81;
        v104[6].m128i_i64[1] += 8;
        goto LABEL_31;
      }
      v104[6].m128i_i64[1] = 8;
    }
    if ( v81 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v81 + 8LL))(v81);
  }
LABEL_31:
  if ( a4 != 1 || !*(_BYTE *)(a2 + 192) || *(_DWORD *)(a2 + 168) != 3 )
  {
    v83 = *(_BYTE *)(a2 + 13);
    v85 = *(_BYTE *)(a2 + 11) ^ 1;
    sub_2332320((__int64)v104, 0, *(unsigned __int8 *)(a2 + 11) ^ 1u, v83, v58, v59);
    v60 = sub_22077B0(0x10u);
    if ( v60 )
    {
      *(_DWORD *)(v60 + 8) = a3;
      *(_QWORD *)v60 = &unk_4A12238;
      *(_BYTE *)(v60 + 12) = v85;
      *(_BYTE *)(v60 + 13) = v83;
    }
    v105.m128i_i64[0] = v60;
    sub_23A46F0(&v104[4].m128i_u64[1], (unsigned __int64 *)&v105);
    sub_233F7D0(v105.m128i_i64);
  }
  sub_23A0E50(a2, (__int64)v104, a3);
  sub_23A20C0((__int64)&v105, (__int64)v97, 1, 1, 0, v61);
  sub_2353940(a1, v105.m128i_i64);
  sub_233F7F0((__int64)&v105.m128i_i64[1]);
  sub_233F7D0(v105.m128i_i64);
  v94 = 0x100010000000001LL;
  v95 = 0x1000101000000LL;
  v96 = 0;
  sub_29744D0(&v105, &v94);
  sub_23A1F80(a1, v105.m128i_i64);
  LOBYTE(v90) = 0;
  HIDWORD(v90) = 1;
  LOBYTE(v91) = 0;
  sub_F10C20((__int64)&v105, v90, v91);
  sub_2353C90(a1, (__int64)&v105, v62, v63, v64, v65);
  sub_233BCC0((__int64)&v105);
  sub_23A20C0((__int64)&v105, (__int64)v104, 0, 0, 0, v66);
  sub_2353940(a1, v105.m128i_i64);
  sub_233F7F0((__int64)&v105.m128i_i64[1]);
  sub_233F7D0(v105.m128i_i64);
  sub_291E720(&v105, 0);
  sub_23A2000(a1, v105.m128i_i8);
  v67 = (_QWORD *)sub_22077B0(0x48u);
  if ( v67 )
  {
    v67[1] = 0;
    v67[2] = 0;
    v67[3] = 0;
    *v67 = &unk_4A10038;
    v67[4] = 0;
    v67[5] = 0;
    v67[6] = 0;
    v67[7] = 0;
    v67[8] = 0;
  }
  v105.m128i_i64[0] = (__int64)v67;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  v68 = (_QWORD *)sub_22077B0(0x10u);
  if ( v68 )
    *v68 = &unk_4A10D38;
  v105.m128i_i64[0] = (__int64)v68;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  v69 = (_QWORD *)sub_22077B0(0x10u);
  if ( v69 )
    *v69 = &unk_4A0EF38;
  v105.m128i_i64[0] = (__int64)v69;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  LOBYTE(v92) = 0;
  HIDWORD(v92) = 1;
  LOBYTE(v93) = 0;
  sub_F10C20((__int64)&v105, v92, v93);
  sub_2353C90(a1, (__int64)&v105, v70, v71, v72, v73);
  sub_233BCC0((__int64)&v105);
  sub_23A0D70(a2, (__int64)a1, a3);
  v74 = (_QWORD *)sub_22077B0(0x10u);
  if ( v74 )
    *v74 = &unk_4A0F178;
  v105.m128i_i64[0] = (__int64)v74;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  sub_23A0EC0(a2, (__int64)a1, a3);
  v75 = (_QWORD *)sub_22077B0(0x10u);
  if ( v75 )
    *v75 = &unk_4A0ED38;
  v105.m128i_i64[0] = (__int64)v75;
  sub_23A1F40(a1, (unsigned __int64 *)&v105);
  sub_233EFE0(v105.m128i_i64);
  v94 = 0x100010000000001LL;
  v95 = 0x1000101000000LL;
  v96 = 0;
  sub_29744D0(&v105, &v94);
  sub_23A1F80(a1, v105.m128i_i64);
  LOBYTE(v94) = 0;
  HIDWORD(v94) = 1;
  LOBYTE(v95) = 0;
  sub_F10C20((__int64)&v105, v94, v95);
  sub_2353C90(a1, (__int64)&v105, v76, v77, v78, v79);
  sub_233BCC0((__int64)&v105);
  sub_23A0D70(a2, (__int64)a1, a3);
  sub_2337B30((unsigned __int64 *)v104);
  sub_2337B30((unsigned __int64 *)v97);
  return a1;
}
