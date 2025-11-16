// Function: sub_32F8D30
// Address: 0x32f8d30
//
__int64 __fastcall sub_32F8D30(__int64 *a1, __int64 a2)
{
  __int64 *v4; // r13
  __m128i v5; // xmm1
  __int16 *v6; // rax
  __int64 v7; // rdx
  __int16 v8; // r14
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 (__fastcall *v11)(__int64 *, __int64, __int64, __int64, __int64); // rbx
  __int64 v12; // rax
  __int16 v13; // ax
  __int64 v14; // rsi
  __int16 v15; // bx
  __int64 v16; // rdx
  __int64 v17; // rdi
  __m128i si128; // xmm3
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rcx
  __int64 v22; // r14
  __int64 v24; // rax
  __int64 v25; // rdi
  unsigned int v26; // edx
  __int64 v27; // rdx
  __int64 v28; // rdi
  __m128i v29; // xmm5
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // rax
  int v33; // r9d
  __int64 v34; // r10
  __int128 v35; // rax
  __int64 v36; // rdi
  int v37; // r9d
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // r15
  __int64 v41; // r8
  __int64 v42; // r9
  __int64 v43; // r8
  __int64 v44; // r9
  char v45; // bl
  char v46; // al
  __int128 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdi
  int v50; // ebx
  int v51; // r12d
  __int64 v52; // r14
  __int64 v53; // rdx
  __int64 v54; // r15
  __int128 v55; // rax
  int v56; // r9d
  __int64 v57; // rax
  int v58; // r12d
  unsigned int v59; // edx
  unsigned int v60; // r8d
  int v61; // edx
  __int32 v62; // r10d
  __int64 v63; // rax
  __int16 v64; // si
  __int64 v65; // rax
  __m128i v66; // rcx
  int v67; // esi
  bool v68; // al
  __int64 v69; // rdi
  __int64 (*v70)(); // r8
  __int128 v71; // [rsp-10h] [rbp-F0h]
  __int64 v72; // [rsp+0h] [rbp-E0h]
  __int64 v73; // [rsp+0h] [rbp-E0h]
  __int64 v74; // [rsp+10h] [rbp-D0h]
  __int128 v75; // [rsp+10h] [rbp-D0h]
  int v76; // [rsp+10h] [rbp-D0h]
  __int64 v77; // [rsp+20h] [rbp-C0h]
  __int128 v78; // [rsp+20h] [rbp-C0h]
  __int64 v79; // [rsp+30h] [rbp-B0h]
  __int64 v80; // [rsp+38h] [rbp-A8h]
  __m128i v81; // [rsp+40h] [rbp-A0h] BYREF
  __int128 v82; // [rsp+50h] [rbp-90h] BYREF
  __int64 v83; // [rsp+60h] [rbp-80h] BYREF
  __int64 v84; // [rsp+68h] [rbp-78h]
  __int64 v85; // [rsp+70h] [rbp-70h] BYREF
  __int64 v86; // [rsp+78h] [rbp-68h]
  __int64 v87; // [rsp+80h] [rbp-60h] BYREF
  int v88; // [rsp+88h] [rbp-58h]
  __m128i v89; // [rsp+90h] [rbp-50h] BYREF
  __m128i v90; // [rsp+A0h] [rbp-40h]

  v4 = (__int64 *)a1[1];
  v5 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 40LL));
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v4;
  v81 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
  v8 = *v6;
  v9 = *((_QWORD *)v6 + 1);
  v82 = (__int128)v5;
  v84 = v9;
  v10 = *a1;
  LOWORD(v83) = v8;
  v11 = *(__int64 (__fastcall **)(__int64 *, __int64, __int64, __int64, __int64))(v7 + 528);
  v74 = *(_QWORD *)(v10 + 64);
  v12 = sub_2E79000(*(__int64 **)(v10 + 40));
  v13 = v11(v4, v12, v74, v83, v84);
  v14 = *(_QWORD *)(a2 + 80);
  LOWORD(v85) = v13;
  v15 = v13;
  v86 = v16;
  v87 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v87, v14, 1);
  v17 = *a1;
  v88 = *(_DWORD *)(a2 + 72);
  si128 = _mm_load_si128((const __m128i *)&v82);
  v89 = _mm_load_si128(&v81);
  v90 = si128;
  v19 = sub_3402EA0(v17, 60, (unsigned int)&v87, v83, v84, 0, (__int64)&v89, 2);
  v21 = v72;
  if ( v19 )
    goto LABEL_4;
  if ( v8 )
  {
    if ( (unsigned __int16)(v8 - 17) > 0xD3u )
      goto LABEL_10;
  }
  else if ( !sub_30070B0((__int64)&v83) )
  {
    goto LABEL_10;
  }
  v19 = sub_3295970(a1, a2, (__int64)&v87, v21, v20);
  if ( v19 )
  {
LABEL_4:
    v22 = v19;
    goto LABEL_5;
  }
LABEL_10:
  v24 = sub_33DFBC0(v82, *((_QWORD *)&v82 + 1), 0, 0);
  v77 = v24;
  if ( v24 )
  {
    v25 = *(_QWORD *)(v24 + 96);
    v26 = *(_DWORD *)(v25 + 32);
    if ( v26 )
    {
      if ( v26 <= 0x40 )
      {
        if ( *(_QWORD *)(v25 + 24) != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) )
          goto LABEL_14;
      }
      else if ( v26 != (unsigned int)sub_C445E0(v25 + 24) )
      {
        goto LABEL_14;
      }
    }
    if ( v15 )
      v45 = (unsigned __int16)(v15 - 17) <= 0xD3u;
    else
      v45 = sub_30070B0((__int64)&v85);
    if ( v8 )
      v46 = (unsigned __int16)(v8 - 17) <= 0xD3u;
    else
      v46 = sub_30070B0((__int64)&v83);
    if ( v46 == v45 )
    {
      v79 = *a1;
      *(_QWORD *)&v47 = sub_3400BD0(*a1, 0, (unsigned int)&v87, v83, v84, 0, 0);
      LODWORD(v73) = 0;
      v78 = v47;
      v48 = sub_3400BD0(*a1, 1, (unsigned int)&v87, v83, v84, 0, v73);
      v49 = *a1;
      v50 = v85;
      v51 = v86;
      v52 = v48;
      v54 = v53;
      *(_QWORD *)&v55 = sub_33ED040(v49, 17);
      v57 = sub_340F900(v49, 208, (unsigned int)&v87, v50, v51, v56, *(_OWORD *)&v81, v82, v55);
      v58 = v84;
      v60 = v59;
      v61 = v83;
      v62 = v79;
      v66.m128i_i64[0] = v57;
      v63 = *(_QWORD *)(v57 + 48) + 16LL * v60;
      v64 = *(_WORD *)v63;
      v65 = *(_QWORD *)(v63 + 8);
      v66.m128i_i64[1] = v60;
      v89.m128i_i16[0] = v64;
      v89.m128i_i64[1] = v65;
      if ( v64 )
      {
        v67 = ((unsigned __int16)(v64 - 17) < 0xD4u) + 205;
      }
      else
      {
        v81.m128i_i64[1] = v60;
        v76 = v83;
        v81.m128i_i64[0] = v66.m128i_i64[0];
        *(_QWORD *)&v82 = v79;
        v68 = sub_30070B0((__int64)&v89);
        v61 = v76;
        v66 = v81;
        v62 = v82;
        v67 = 205 - (!v68 - 1);
      }
      *((_QWORD *)&v71 + 1) = v54;
      *(_QWORD *)&v71 = v52;
      v22 = sub_340EC60(v62, v67, (unsigned int)&v87, v61, v58, 0, v66.m128i_i64[0], v66.m128i_i64[1], v71, v78);
      goto LABEL_5;
    }
  }
LABEL_14:
  v19 = sub_3269740(a2, *a1);
  if ( v19 )
    goto LABEL_4;
  v19 = sub_329BF20(a1, a2);
  if ( v19 )
    goto LABEL_4;
  *(_QWORD *)&v75 = sub_32CA770(a1, v81.m128i_i64[0], v81.m128i_i64[1], v82, *((__int64 *)&v82 + 1), a2);
  *((_QWORD *)&v75 + 1) = v27;
  if ( (_QWORD)v75 )
  {
    v28 = *a1;
    v29 = _mm_load_si128((const __m128i *)&v82);
    v30 = *(_QWORD *)(a2 + 48);
    v31 = *(unsigned int *)(a2 + 68);
    v89 = _mm_load_si128(&v81);
    v90 = v29;
    v32 = sub_33D01C0(v28, 62, v30, v31, &v89, 2);
    v34 = v75;
    v80 = v32;
    if ( v32 )
    {
      *(_QWORD *)&v35 = sub_3406EB0(*a1, 58, (unsigned int)&v87, v83, v84, v33, v75, v82);
      v36 = *a1;
      v82 = v35;
      v38 = sub_3406EB0(v36, 57, (unsigned int)&v87, v83, v84, v37, *(_OWORD *)&v81, v35);
      v40 = v39;
      sub_32B3E80((__int64)a1, v82, 1, 0, v41, v42);
      sub_32B3E80((__int64)a1, v38, 1, 0, v43, v44);
      v89.m128i_i64[0] = v38;
      v89.m128i_i64[1] = v40;
      sub_32EB790((__int64)a1, v80, v89.m128i_i64, 1, 1);
      v34 = v75;
    }
    v22 = v34;
  }
  else if ( v77
         && ((v69 = a1[1], v70 = *(__int64 (**)())(*(_QWORD *)v69 + 200LL), v70 == sub_2FE2F30)
          || !((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, _QWORD))v70)(
                v69,
                **(unsigned __int16 **)(a2 + 48),
                *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
                *(_QWORD *)(**(_QWORD **)(*a1 + 40) + 120LL)))
         || (v22 = sub_32EC020(a1, a2)) == 0 )
  {
    if ( (unsigned __int8)sub_32D0FE0((__int64)a1, a2, 0) )
      v22 = a2;
    else
      v22 = 0;
  }
LABEL_5:
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
  return v22;
}
