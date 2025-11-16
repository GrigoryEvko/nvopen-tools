// Function: sub_3469F50
// Address: 0x3469f50
//
void __fastcall sub_3469F50(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 **a4, __int64 a5)
{
  __int64 v8; // rsi
  __int64 *v9; // rdx
  __int64 v10; // r9
  unsigned __int16 *v11; // rsi
  __m128i v12; // xmm0
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rax
  unsigned int v16; // r14d
  __int64 v17; // rdx
  unsigned __int8 *v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int16 *v21; // rax
  __int64 v22; // r14
  __int64 v23; // r15
  __int64 v24; // rax
  unsigned int v25; // eax
  unsigned int v26; // esi
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // r13
  __int128 v30; // rax
  __int64 v31; // r9
  unsigned int v32; // edx
  __int64 v33; // rsi
  __int64 v34; // r9
  __int64 v35; // r15
  unsigned __int8 *v36; // rax
  unsigned __int8 **v37; // rbx
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned __int8 *v40; // rax
  unsigned int *v41; // rcx
  __m128i v42; // xmm2
  __int64 v43; // rdx
  __m128i v44; // xmm3
  __int64 v45; // r8
  __int64 v46; // r9
  unsigned __int8 *v47; // rax
  __int64 v48; // rbx
  __int64 v49; // rsi
  unsigned __int8 **v50; // rbx
  unsigned __int8 *v51; // r12
  __int64 v52; // rdx
  __int64 v53; // r13
  __int128 v54; // rax
  __int64 v55; // r9
  __int64 v56; // rax
  unsigned __int8 *v57; // r12
  __int64 v58; // rdx
  __int64 v59; // r13
  __int64 v60; // r9
  __int128 v61; // rax
  __int128 v62; // [rsp-30h] [rbp-130h]
  __int128 v63; // [rsp-30h] [rbp-130h]
  __int128 v64; // [rsp-30h] [rbp-130h]
  __int128 v65; // [rsp-20h] [rbp-120h]
  __int128 v66; // [rsp-10h] [rbp-110h]
  int v67; // [rsp+Ch] [rbp-F4h]
  __int64 (__fastcall *v68)(__int64, __int64, __int64, __int64, __int64); // [rsp+10h] [rbp-F0h]
  __int64 v69; // [rsp+18h] [rbp-E8h]
  __int64 v70; // [rsp+20h] [rbp-E0h]
  unsigned int v71; // [rsp+20h] [rbp-E0h]
  __int64 v72; // [rsp+28h] [rbp-D8h]
  __int64 v73; // [rsp+28h] [rbp-D8h]
  __m128i v74; // [rsp+30h] [rbp-D0h] BYREF
  __m128i v75; // [rsp+40h] [rbp-C0h] BYREF
  __int128 v76; // [rsp+50h] [rbp-B0h]
  unsigned __int8 **v77; // [rsp+60h] [rbp-A0h]
  __int64 *v78; // [rsp+68h] [rbp-98h]
  unsigned __int8 *v79; // [rsp+70h] [rbp-90h]
  __int64 v80; // [rsp+78h] [rbp-88h]
  unsigned __int8 *v81; // [rsp+80h] [rbp-80h]
  __int64 v82; // [rsp+88h] [rbp-78h]
  __int64 v83; // [rsp+90h] [rbp-70h] BYREF
  int v84; // [rsp+98h] [rbp-68h]
  _OWORD v85[2]; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int8 *v86; // [rsp+C0h] [rbp-40h]
  __int64 v87; // [rsp+C8h] [rbp-38h]

  v8 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v76 = a3;
  v77 = a4;
  v83 = v8;
  v78 = &v83;
  if ( v8 )
    sub_B96E90((__int64)&v83, v8, 1);
  v9 = *(__int64 **)(a2 + 40);
  v10 = *(unsigned int *)(a2 + 24);
  v11 = *(unsigned __int16 **)(a2 + 48);
  v84 = *(_DWORD *)(a2 + 72);
  v12 = _mm_loadu_si128((const __m128i *)v9);
  v13 = *v11;
  v14 = *v9;
  v15 = *((unsigned int *)v9 + 2);
  v74 = _mm_loadu_si128((const __m128i *)(v9 + 5));
  v75 = v12;
  v16 = ((_DWORD)v10 != 77) + 72;
  v17 = 1;
  if ( (_WORD)v13 != 1 && (!(_WORD)v13 || (v17 = (unsigned __int16)v13, !*(_QWORD *)(a1 + 8 * v13 + 112)))
    || (*(_BYTE *)(v16 + a1 + 500 * v17 + 6414) & 0xFB) != 0 )
  {
    v67 = v10;
    v18 = sub_3406EB0(
            (_QWORD *)a5,
            (unsigned int)((_DWORD)v10 != 77) + 56,
            (__int64)v78,
            *(unsigned __int16 *)(*(_QWORD *)(v14 + 48) + 16 * v15),
            *(_QWORD *)(*(_QWORD *)(v14 + 48) + 16 * v15 + 8),
            v10,
            *(_OWORD *)&v75,
            *(_OWORD *)&v74);
    v19 = v76;
    v81 = v18;
    v82 = v20;
    *(_QWORD *)v76 = v18;
    *(_DWORD *)(v19 + 8) = v82;
    v21 = *(unsigned __int16 **)(a2 + 48);
    v22 = v21[8];
    v23 = *((_QWORD *)v21 + 3);
    v69 = *((_QWORD *)v21 + 1);
    v68 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 528LL);
    v70 = *v21;
    v72 = *(_QWORD *)(a5 + 64);
    v24 = sub_2E79000(*(__int64 **)(a5 + 40));
    v25 = v68(a1, v24, v72, v70, v69);
    v26 = 10;
    v73 = v27;
    v71 = v25;
    if ( v67 != 77 )
    {
LABEL_6:
      v28 = *(_QWORD *)v76;
      v29 = *(_QWORD *)(v76 + 8);
      *(_QWORD *)&v30 = sub_33ED040((_QWORD *)a5, v26);
      *((_QWORD *)&v62 + 1) = v29;
      *(_QWORD *)&v62 = v28;
      v33 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)v78, v71, v73, v31, v62, *(_OWORD *)&v75, v30);
      goto LABEL_7;
    }
    if ( sub_33CF4D0(v74.m128i_i64[0]) )
    {
      v57 = sub_3400BD0(
              a5,
              0,
              (__int64)v78,
              **(unsigned __int16 **)(a2 + 48),
              *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
              0,
              v12,
              0);
      v59 = v58;
      v60 = *(_QWORD *)(v76 + 8);
      *(_QWORD *)&v76 = *(_QWORD *)v76;
      *((_QWORD *)&v76 + 1) = v60;
      *(_QWORD *)&v61 = sub_33ED040((_QWORD *)a5, 0x11u);
      *((_QWORD *)&v64 + 1) = v59;
      *(_QWORD *)&v64 = v57;
      v56 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)v78, v71, v73, *((__int64 *)&v76 + 1), v76, v64, v61);
    }
    else
    {
      v26 = 12;
      if ( !sub_33CF460(v74.m128i_i64[0]) )
        goto LABEL_6;
      v51 = sub_3400BD0(
              a5,
              0,
              (__int64)v78,
              **(unsigned __int16 **)(a2 + 48),
              *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
              0,
              v12,
              0);
      v53 = v52;
      *(_QWORD *)&v54 = sub_33ED040((_QWORD *)a5, 0x16u);
      *((_QWORD *)&v63 + 1) = v53;
      *(_QWORD *)&v63 = v51;
      v56 = sub_340F900((_QWORD *)a5, 0xD0u, (__int64)v78, v71, v73, v55, *(_OWORD *)&v75, v63, v54);
    }
    v33 = v56;
LABEL_7:
    *((_QWORD *)&v66 + 1) = v23;
    v34 = v23;
    v35 = (__int64)v78;
    *(_QWORD *)&v66 = v22;
    v36 = sub_33FB620(a5, v33, v32, (__int64)v78, v22, v34, v12, v66);
    v37 = v77;
    v38 = v83;
    v80 = v39;
    v79 = v36;
    *v77 = v36;
    *((_DWORD *)v37 + 2) = v80;
    if ( v38 )
      sub_B91220(v35, v38);
    return;
  }
  v40 = sub_3400BD0(a5, 0, (__int64)v78, v11[8], *((_QWORD *)v11 + 3), 0, v12, 0);
  v41 = *(unsigned int **)(a2 + 48);
  v86 = v40;
  v42 = _mm_load_si128(&v75);
  v87 = v43;
  v44 = _mm_load_si128(&v74);
  *((_QWORD *)&v65 + 1) = 3;
  *(_QWORD *)&v65 = v85;
  v45 = *(unsigned int *)(a2 + 68);
  v85[0] = v42;
  v85[1] = v44;
  v47 = sub_3411630((_QWORD *)a5, v16, (__int64)v78, v41, v45, v46, v65);
  v48 = v76;
  v49 = v83;
  *(_QWORD *)v76 = v47;
  *(_DWORD *)(v48 + 8) = 0;
  v50 = v77;
  *v77 = v47;
  *((_DWORD *)v50 + 2) = 1;
  if ( v49 )
    sub_B91220((__int64)v78, v49);
}
