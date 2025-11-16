// Function: sub_3846160
// Address: 0x3846160
//
void __fastcall sub_3846160(__int64 *a1, __int64 a2, __m128i *a3, __int64 a4, __int64 a5, __m128i a6)
{
  unsigned int v6; // r15d
  unsigned __int16 *v8; // r12
  int v9; // eax
  __int64 v10; // rdx
  __int64 v11; // rcx
  int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // r11
  __int64 v16; // rdx
  __int64 v17; // rax
  __int64 v18; // r9
  __int64 v19; // r8
  unsigned __int16 v20; // r10
  __int64 (__fastcall *v21)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v22; // r10
  __int64 *v23; // rax
  int v24; // esi
  bool v25; // zf
  int v26; // eax
  __int64 v27; // r9
  __int64 v28; // r8
  __int128 v29; // rax
  __int64 v30; // r11
  __int64 v31; // r9
  unsigned int v32; // edx
  __int64 v33; // r15
  __int64 v34; // r9
  __int32 v35; // edx
  __int128 v36; // rax
  __int64 v37; // r9
  unsigned __int8 *v38; // rax
  unsigned int v39; // edx
  __int64 v40; // r9
  int v41; // edx
  __m128i v42; // xmm0
  unsigned int v43; // r14d
  __int64 *v44; // rax
  int v45; // eax
  __int64 v46; // r9
  __int64 v47; // r8
  unsigned int v48; // esi
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rdx
  unsigned int v52; // eax
  __int64 v53; // rdx
  __int128 v54; // [rsp-30h] [rbp-150h]
  __int128 v55; // [rsp-20h] [rbp-140h]
  __int128 v56; // [rsp-10h] [rbp-130h]
  __int16 v57; // [rsp+2h] [rbp-11Eh]
  __int64 v58; // [rsp+8h] [rbp-118h]
  __int64 v59; // [rsp+10h] [rbp-110h]
  __int64 *v61; // [rsp+20h] [rbp-100h]
  unsigned __int16 v62; // [rsp+28h] [rbp-F8h]
  __int16 v63; // [rsp+2Ch] [rbp-F4h]
  unsigned __int8 v64; // [rsp+2Fh] [rbp-F1h]
  int v65; // [rsp+30h] [rbp-F0h]
  __int64 *v66; // [rsp+30h] [rbp-F0h]
  _QWORD *v67; // [rsp+30h] [rbp-F0h]
  __int64 v68; // [rsp+38h] [rbp-E8h]
  unsigned __int8 v69; // [rsp+40h] [rbp-E0h]
  __int64 v70; // [rsp+40h] [rbp-E0h]
  __int64 v71; // [rsp+40h] [rbp-E0h]
  __int128 v72; // [rsp+40h] [rbp-E0h]
  __int64 v73; // [rsp+40h] [rbp-E0h]
  __int64 v74; // [rsp+48h] [rbp-D8h]
  unsigned int v76; // [rsp+58h] [rbp-C8h]
  __int128 v77; // [rsp+60h] [rbp-C0h]
  __int64 v78; // [rsp+B8h] [rbp-68h]
  __int64 v79; // [rsp+C0h] [rbp-60h] BYREF
  int v80; // [rsp+C8h] [rbp-58h]
  __int64 v81; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v82; // [rsp+D8h] [rbp-48h]
  __int64 v83; // [rsp+E0h] [rbp-40h]

  v8 = (unsigned __int16 *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL));
  v9 = *v8;
  v10 = *((_QWORD *)v8 + 1);
  LOWORD(v81) = v9;
  v82 = v10;
  if ( (_WORD)v9 )
  {
    v69 = (unsigned __int16)(v9 - 176) <= 0x34u;
    v11 = v69;
    v64 = v69;
    v65 = word_4456340[v9 - 1];
  }
  else
  {
    v78 = sub_3007240((__int64)&v81);
    v65 = v78;
    v64 = BYTE4(v78);
    v69 = BYTE4(v78);
  }
  v12 = *v8;
  v13 = *((_QWORD *)v8 + 1);
  LOWORD(v81) = v12;
  v82 = v13;
  if ( (_WORD)v12 )
  {
    v58 = 0;
    v63 = word_4456580[v12 - 1];
  }
  else
  {
    v63 = sub_3009970((__int64)&v81, a2, v13, v11, a5);
    v58 = v51;
  }
  v14 = *(_QWORD *)(a2 + 80);
  v79 = v14;
  if ( v14 )
    sub_B96E90((__int64)&v79, v14, 1);
  v15 = *a1;
  v16 = a1[1];
  v80 = *(_DWORD *)(a2 + 72);
  v17 = *(_QWORD *)(a2 + 48);
  v18 = *(_QWORD *)(v16 + 64);
  v19 = *(_QWORD *)(v17 + 8);
  v20 = *(_WORD *)v17;
  v59 = v19;
  v21 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v15 + 592LL);
  v62 = v20;
  if ( v21 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v81, v15, v18, v20, v19);
    v22 = v62;
    v76 = (unsigned __int16)v82;
    v68 = v83;
  }
  else
  {
    v52 = v21(v15, v18, v20, v19);
    v22 = v62;
    v76 = v52;
    v68 = v53;
  }
  if ( v63 != v22 || !v22 && v59 != v58 )
  {
    v43 = v22;
    v44 = *(__int64 **)(a1[1] + 64);
    LODWORD(v81) = v65;
    v61 = v44;
    BYTE4(v81) = v64;
    if ( v69 )
    {
      LOWORD(v45) = sub_2D43AD0(v22, v65);
      v47 = 0;
      if ( (_WORD)v45 )
        goto LABEL_21;
    }
    else
    {
      LOWORD(v45) = sub_2D43050(v22, v65);
      v47 = 0;
      if ( (_WORD)v45 )
      {
LABEL_21:
        HIWORD(v48) = v57;
        LOWORD(v48) = v45;
        sub_33FAF80(a1[1], 215, (__int64)&v79, v48, v47, v46, a6);
        goto LABEL_12;
      }
    }
    v45 = sub_3009450(v61, v43, v59, v81, 0, v46);
    v57 = HIWORD(v45);
    v47 = v50;
    goto LABEL_21;
  }
LABEL_12:
  v23 = *(__int64 **)(a1[1] + 64);
  v24 = 2 * v65;
  v25 = v69 == 0;
  v70 = a1[1];
  LODWORD(v81) = 2 * v65;
  v66 = v23;
  BYTE4(v81) = v64;
  if ( v25 )
  {
    LOWORD(v26) = sub_2D43050(v76, v24);
    v27 = v70;
    v28 = 0;
    if ( (_WORD)v26 )
      goto LABEL_14;
  }
  else
  {
    LOWORD(v26) = sub_2D43AD0(v76, v24);
    v27 = v70;
    v28 = 0;
    if ( (_WORD)v26 )
      goto LABEL_14;
  }
  v73 = v27;
  v26 = sub_3009450(v66, v76, v68, v81, 0, v27);
  v27 = v73;
  HIWORD(v6) = HIWORD(v26);
  v28 = v49;
LABEL_14:
  LOWORD(v6) = v26;
  *(_QWORD *)&v29 = sub_33FAF80(v27, 234, (__int64)&v79, v6, v28, v27, a6);
  v77 = v29;
  *(_QWORD *)&v29 = *(_QWORD *)(a2 + 40);
  v30 = *(_QWORD *)(v29 + 48);
  v71 = *(_QWORD *)(v29 + 40);
  *(_QWORD *)&v29 = *(_QWORD *)(v71 + 48) + 16LL * *(unsigned int *)(v29 + 48);
  v74 = v30;
  *((_QWORD *)&v55 + 1) = v30;
  *(_QWORD *)&v55 = v71;
  *((_QWORD *)&v54 + 1) = v30;
  *(_QWORD *)&v54 = v71;
  *(_QWORD *)&v72 = sub_3406EB0(
                      (_QWORD *)a1[1],
                      0x38u,
                      (__int64)&v79,
                      *(unsigned __int16 *)v29,
                      *(_QWORD *)(v29 + 8),
                      v31,
                      v54,
                      v55);
  v33 = 16LL * v32;
  *((_QWORD *)&v72 + 1) = v32 | v74 & 0xFFFFFFFF00000000LL;
  a3->m128i_i64[0] = (__int64)sub_3406EB0((_QWORD *)a1[1], 0x9Eu, (__int64)&v79, v76, v68, v34, v77, v72);
  a3->m128i_i32[2] = v35;
  v67 = (_QWORD *)a1[1];
  *(_QWORD *)&v36 = sub_3400BD0(
                      (__int64)v67,
                      1,
                      (__int64)&v79,
                      *(unsigned __int16 *)(v33 + *(_QWORD *)(v72 + 48)),
                      *(_QWORD *)(v33 + *(_QWORD *)(v72 + 48) + 8),
                      0,
                      a6,
                      0);
  v38 = sub_3406EB0(
          v67,
          0x38u,
          (__int64)&v79,
          *(unsigned __int16 *)(*(_QWORD *)(v72 + 48) + v33),
          *(_QWORD *)(*(_QWORD *)(v72 + 48) + v33 + 8),
          v37,
          v72,
          v36);
  *((_QWORD *)&v56 + 1) = v39 | *((_QWORD *)&v72 + 1) & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v56 = v38;
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)a1[1], 0x9Eu, (__int64)&v79, v76, v68, v40, v77, v56);
  *(_DWORD *)(a4 + 8) = v41;
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(a1[1] + 40)) )
  {
    v42 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = v42.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v42.m128i_i32[2];
  }
  if ( v79 )
    sub_B91220((__int64)&v79, v79);
}
