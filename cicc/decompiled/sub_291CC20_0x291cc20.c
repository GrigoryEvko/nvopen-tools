// Function: sub_291CC20
// Address: 0x291cc20
//
unsigned __int64 __fastcall sub_291CC20(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        const __m128i *a6)
{
  _BYTE *v6; // r10
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rbx
  __int64 v10; // rax
  __int64 v11; // r14
  unsigned __int64 result; // rax
  __int64 v13; // r8
  __int64 v14; // r9
  __int32 v15; // eax
  unsigned __int32 v16; // edi
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  unsigned __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rdi
  unsigned __int8 *v23; // rbx
  __int64 (__fastcall *v24)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  unsigned __int8 *v28; // r14
  __int64 v29; // rdi
  __int64 (__fastcall *v30)(__int64, unsigned int, unsigned __int8 *, _BYTE *); // rax
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rax
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // rdi
  __int64 (__fastcall *v44)(__int64, _BYTE *, _BYTE *, unsigned __int8 *); // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // r10
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int8 *v51; // r10
  __int64 (__fastcall *v52)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v53; // rax
  __int64 v54; // rdx
  unsigned __int64 v55; // r14
  __int64 v56; // rdx
  __int64 v57; // rbx
  __int64 v58; // r12
  __int64 v59; // rdx
  unsigned int v60; // esi
  __int64 v61; // rbx
  __int64 v62; // r12
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // rbx
  __int64 v66; // r12
  __int64 v67; // r13
  __int64 v68; // rdx
  unsigned int v69; // esi
  _QWORD *v70; // rax
  __int64 v71; // r9
  _QWORD *v72; // r10
  __int64 v73; // rsi
  __int64 v74; // rax
  __int64 v75; // rbx
  __int64 v76; // r13
  __int64 v77; // r12
  __int64 v78; // rdx
  unsigned int v79; // esi
  __int64 v80; // [rsp-8h] [rbp-118h]
  unsigned __int64 v81; // [rsp+8h] [rbp-108h]
  _BYTE *v84; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v85; // [rsp+20h] [rbp-F0h]
  __int64 v86; // [rsp+20h] [rbp-F0h]
  __int64 v87; // [rsp+20h] [rbp-F0h]
  __int64 v88; // [rsp+20h] [rbp-F0h]
  unsigned __int8 *v89; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v91; // [rsp+30h] [rbp-E0h]
  unsigned __int8 *v92; // [rsp+30h] [rbp-E0h]
  __int64 v93; // [rsp+30h] [rbp-E0h]
  __int64 v94; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v95; // [rsp+38h] [rbp-D8h]
  __int64 **v96; // [rsp+38h] [rbp-D8h]
  _BYTE *v97; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v98; // [rsp+38h] [rbp-D8h]
  __int64 v99; // [rsp+38h] [rbp-D8h]
  __int64 v100; // [rsp+38h] [rbp-D8h]
  _QWORD *v101; // [rsp+38h] [rbp-D8h]
  __int64 v102; // [rsp+38h] [rbp-D8h]
  __int64 v103; // [rsp+38h] [rbp-D8h]
  unsigned __int64 v104; // [rsp+40h] [rbp-D0h] BYREF
  unsigned int v105; // [rsp+48h] [rbp-C8h]
  __m128i v106[2]; // [rsp+50h] [rbp-C0h] BYREF
  char v107; // [rsp+70h] [rbp-A0h]
  char v108; // [rsp+71h] [rbp-9Fh]
  __m128i v109[2]; // [rsp+80h] [rbp-90h] BYREF
  char v110; // [rsp+A0h] [rbp-70h]
  char v111; // [rsp+A1h] [rbp-6Fh]
  __m128i v112[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int16 v113; // [rsp+D0h] [rbp-40h]

  v6 = (_BYTE *)a1;
  v7 = a2;
  v8 = a4;
  v91 = a5;
  v9 = *(_QWORD *)(a3 + 8);
  v94 = *(_QWORD *)(a4 + 8);
  if ( (_BYTE)qword_5005508 )
  {
    v31 = sub_9208B0(a1, *(_QWORD *)(a4 + 8));
    v112[0].m128i_i64[1] = v32;
    v112[0].m128i_i64[0] = (unsigned __int64)(v31 + 7) >> 3;
    v81 = sub_CA1930(v112);
    v33 = sub_9208B0(a1, v9);
    v112[0].m128i_i64[1] = v34;
    v112[0].m128i_i64[0] = (unsigned __int64)(v33 + 7) >> 3;
    v35 = sub_CA1930(v112);
    v6 = (_BYTE *)a1;
    a5 = v35;
    if ( v35 == 2 * v81 && (!v91 || v81 == v91) )
    {
      v36 = sub_BCDA70((__int64 *)v94, 2);
      v111 = 1;
      v96 = (__int64 **)v36;
      v109[0].m128i_i64[0] = (__int64)".castvec";
      v110 = 3;
      sub_9C6370(v112, a6, v109, v37, v38, v39);
      v97 = (_BYTE *)sub_291AC80((__int64 *)a2, 0x31u, a3, v96, (__int64)v112, 0, v106[0].m128i_i32[0], 0);
      v40 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
      v92 = (unsigned __int8 *)sub_ACD640(v40, (unsigned int)(v91 / v81), 0);
      v108 = 1;
      v106[0].m128i_i64[0] = (__int64)".insert";
      v107 = 3;
      sub_9C6370(v109, a6, v106, (__int64)".insert", v41, v42);
      v43 = *(_QWORD *)(a2 + 80);
      v44 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, unsigned __int8 *))(*(_QWORD *)v43 + 104LL);
      if ( v44 == sub_948040 )
      {
        if ( *v97 > 0x15u || *(_BYTE *)v8 > 0x15u || *v92 > 0x15u )
          goto LABEL_77;
        v48 = sub_AD5A90((__int64)v97, (_BYTE *)v8, v92, 0);
      }
      else
      {
        v48 = v44(v43, v97, (_BYTE *)v8, v92);
      }
      if ( v48 )
      {
LABEL_47:
        v98 = v48;
        v109[0].m128i_i64[0] = (__int64)".castback";
        v111 = 1;
        v110 = 3;
        sub_9C6370(v112, a6, v109, v45, v46, v47);
        return sub_291AC80((__int64 *)v7, 0x31u, v98, (__int64 **)v9, (__int64)v112, 0, v106[0].m128i_i32[0], 0);
      }
LABEL_77:
      v113 = 257;
      v70 = sub_BD2C40(72, 3u);
      v71 = 0;
      v72 = v70;
      if ( v70 )
      {
        v73 = (__int64)v97;
        v101 = v70;
        sub_B4DFA0((__int64)v70, v73, v8, (__int64)v92, (__int64)v112, 0, 0, 0);
        v72 = v101;
        v71 = v80;
      }
      v102 = (__int64)v72;
      (*(void (__fastcall **)(_QWORD, _QWORD *, __m128i *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v7 + 88) + 16LL))(
        *(_QWORD *)(v7 + 88),
        v72,
        v109,
        *(_QWORD *)(v7 + 56),
        *(_QWORD *)(v7 + 64),
        v71);
      v48 = v102;
      v74 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
      if ( *(_QWORD *)v7 != v74 )
      {
        v103 = v9;
        v75 = v48;
        v93 = v7;
        v76 = *(_QWORD *)v7;
        v77 = v74;
        do
        {
          v78 = *(_QWORD *)(v76 + 8);
          v79 = *(_DWORD *)v76;
          v76 += 16;
          sub_B99FD0(v75, v79, v78);
        }
        while ( v77 != v76 );
        v48 = v75;
        v7 = v93;
        v9 = v103;
      }
      goto LABEL_47;
    }
  }
  if ( v9 != v94 )
  {
    v84 = v6;
    v111 = 1;
    v109[0].m128i_i64[0] = (__int64)".ext";
    v110 = 3;
    sub_9C6370(v112, a6, v109, a4, a5, (__int64)a6);
    v10 = sub_A82F30((unsigned int **)a2, v8, v9, (__int64)v112, 0);
    v6 = v84;
    v8 = v10;
  }
  if ( *v6 )
  {
    v86 = (__int64)v6;
    v112[0].m128i_i64[0] = sub_9208B0((__int64)v6, v9);
    v112[0].m128i_i64[1] = v54;
    v55 = v112[0].m128i_i64[0] + 7;
    v112[0].m128i_i64[0] = sub_9208B0(v86, v94);
    v112[0].m128i_i64[1] = v56;
    v11 = 8 * ((v55 >> 3) - ((unsigned __int64)(v112[0].m128i_i64[0] + 7) >> 3) - v91);
  }
  else
  {
    v11 = 8 * v91;
  }
  if ( v11 )
  {
    v108 = 1;
    v106[0].m128i_i64[0] = (__int64)".shift";
    v107 = 3;
    sub_9C6370(v109, a6, v106, a4, a5, (__int64)a6);
    v49 = sub_AD64C0(*(_QWORD *)(v8 + 8), v11, 0);
    v50 = *(_QWORD *)(a2 + 80);
    v51 = (unsigned __int8 *)v49;
    v52 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))(*(_QWORD *)v50 + 32LL);
    if ( v52 == sub_9201A0 )
    {
      if ( *(_BYTE *)v8 > 0x15u || *v51 > 0x15u )
        goto LABEL_69;
      v85 = v51;
      if ( (unsigned __int8)sub_AC47B0(25) )
        v53 = sub_AD5570(25, v8, v85, 0, 0);
      else
        v53 = sub_AABE40(0x19u, (unsigned __int8 *)v8, v85);
      v51 = v85;
    }
    else
    {
      v89 = v51;
      v53 = v52(v50, 25u, (_BYTE *)v8, v51, 0, 0);
      v51 = v89;
    }
    if ( v53 )
    {
LABEL_55:
      v8 = v53;
      goto LABEL_8;
    }
LABEL_69:
    v113 = 257;
    v87 = sub_B504D0(25, v8, (__int64)v51, (__int64)v112, 0, 0);
    (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v87,
      v109,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v53 = v87;
    if ( *(_QWORD *)a2 != *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8) )
    {
      v88 = v9;
      v65 = *(_QWORD *)a2;
      v66 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
      v67 = v53;
      do
      {
        v68 = *(_QWORD *)(v65 + 8);
        v69 = *(_DWORD *)v65;
        v65 += 16;
        sub_B99FD0(v67, v69, v68);
      }
      while ( v66 != v65 );
      v53 = v67;
      v9 = v88;
      v7 = a2;
    }
    goto LABEL_55;
  }
  result = v8;
  if ( *(_DWORD *)(v94 + 8) >> 8 >= *(_DWORD *)(v9 + 8) >> 8 )
    return result;
LABEL_8:
  sub_BCB300((__int64)v106, v94);
  sub_C449B0((__int64)v109, (const void **)v106, *(_DWORD *)(v9 + 8) >> 8);
  v15 = v109[0].m128i_i32[2];
  v112[0].m128i_i32[2] = v109[0].m128i_i32[2];
  if ( v109[0].m128i_i32[2] <= 0x40u )
  {
    v16 = v109[0].m128i_u32[2];
    v112[0].m128i_i64[0] = v109[0].m128i_i64[0];
LABEL_10:
    v17 = 0;
    if ( (_DWORD)v11 != v15 )
      v17 = v112[0].m128i_i64[0] << v11;
    v18 = ~((0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & v17);
    if ( !v15 )
      v18 = -1;
    goto LABEL_14;
  }
  sub_C43780((__int64)v112, (const void **)v109);
  v15 = v112[0].m128i_i32[2];
  if ( v112[0].m128i_i32[2] <= 0x40u )
  {
    v16 = v109[0].m128i_u32[2];
    goto LABEL_10;
  }
  sub_C47690(v112[0].m128i_i64, v11);
  v15 = v112[0].m128i_i32[2];
  if ( v112[0].m128i_i32[2] > 0x40u )
  {
    sub_C43D10((__int64)v112);
    v15 = v112[0].m128i_i32[2];
    v20 = v112[0].m128i_i64[0];
    v16 = v109[0].m128i_u32[2];
    goto LABEL_16;
  }
  v16 = v109[0].m128i_u32[2];
  v18 = ~v112[0].m128i_i64[0];
LABEL_14:
  v19 = 0;
  v20 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & v18;
  if ( !v15 )
    v20 = 0;
LABEL_16:
  v105 = v15;
  v104 = v20;
  if ( v16 > 0x40 && v109[0].m128i_i64[0] )
    j_j___libc_free_0_0(v109[0].m128i_u64[0]);
  if ( v106[0].m128i_i32[2] > 0x40u && v106[0].m128i_i64[0] )
    j_j___libc_free_0_0(v106[0].m128i_u64[0]);
  v106[0].m128i_i64[0] = (__int64)".mask";
  v108 = 1;
  v107 = 3;
  sub_9C6370(v109, a6, v106, v19, v13, v14);
  v21 = sub_AD8D80(*(_QWORD *)(a3 + 8), (__int64)&v104);
  v22 = *(_QWORD *)(v7 + 80);
  v23 = (unsigned __int8 *)v21;
  v24 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v22 + 16LL);
  if ( v24 != sub_9202E0 )
  {
    v28 = (unsigned __int8 *)v24(v22, 28u, (_BYTE *)a3, v23);
    goto LABEL_27;
  }
  if ( *(_BYTE *)a3 <= 0x15u && *v23 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(28) )
      v28 = (unsigned __int8 *)sub_AD5570(28, a3, v23, 0, 0);
    else
      v28 = (unsigned __int8 *)sub_AABE40(0x1Cu, (unsigned __int8 *)a3, v23);
LABEL_27:
    if ( v28 )
      goto LABEL_28;
  }
  v113 = 257;
  v28 = (unsigned __int8 *)sub_B504D0(28, a3, (__int64)v23, (__int64)v112, 0, 0);
  (*(void (__fastcall **)(_QWORD, unsigned __int8 *, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
    *(_QWORD *)(v7 + 88),
    v28,
    v109,
    *(_QWORD *)(v7 + 56),
    *(_QWORD *)(v7 + 64));
  v25 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
  if ( *(_QWORD *)v7 != v25 )
  {
    v99 = v8;
    v57 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
    v58 = *(_QWORD *)v7;
    do
    {
      v59 = *(_QWORD *)(v58 + 8);
      v60 = *(_DWORD *)v58;
      v58 += 16;
      sub_B99FD0((__int64)v28, v60, v59);
    }
    while ( v57 != v58 );
    v8 = v99;
  }
LABEL_28:
  v106[0].m128i_i64[0] = (__int64)".insert";
  v108 = 1;
  v107 = 3;
  sub_9C6370(v109, a6, v106, v25, v26, v27);
  v29 = *(_QWORD *)(v7 + 80);
  v30 = *(__int64 (__fastcall **)(__int64, unsigned int, unsigned __int8 *, _BYTE *))(*(_QWORD *)v29 + 16LL);
  if ( v30 == sub_9202E0 )
  {
    if ( *v28 > 0x15u || *(_BYTE *)v8 > 0x15u )
    {
LABEL_61:
      v113 = 257;
      v100 = sub_B504D0(29, (__int64)v28, v8, (__int64)v112, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD))(**(_QWORD **)(v7 + 88) + 16LL))(
        *(_QWORD *)(v7 + 88),
        v100,
        v109,
        *(_QWORD *)(v7 + 56),
        *(_QWORD *)(v7 + 64));
      v61 = *(_QWORD *)v7;
      result = v100;
      v62 = *(_QWORD *)v7 + 16LL * *(unsigned int *)(v7 + 8);
      if ( *(_QWORD *)v7 != v62 )
      {
        do
        {
          v63 = *(_QWORD *)(v61 + 8);
          v64 = *(_DWORD *)v61;
          v61 += 16;
          sub_B99FD0(v100, v64, v63);
        }
        while ( v62 != v61 );
        result = v100;
      }
      goto LABEL_34;
    }
    if ( (unsigned __int8)sub_AC47B0(29) )
      result = sub_AD5570(29, (__int64)v28, (unsigned __int8 *)v8, 0, 0);
    else
      result = sub_AABE40(0x1Du, v28, (unsigned __int8 *)v8);
  }
  else
  {
    result = v30(v29, 29u, v28, (_BYTE *)v8);
  }
  if ( !result )
    goto LABEL_61;
LABEL_34:
  if ( v105 > 0x40 )
  {
    if ( v104 )
    {
      v95 = result;
      j_j___libc_free_0_0(v104);
      return v95;
    }
  }
  return result;
}
