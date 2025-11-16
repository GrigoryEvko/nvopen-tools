// Function: sub_36E30C0
// Address: 0x36e30c0
//
__int64 __fastcall sub_36E30C0(__int64 a1, __int64 a2)
{
  int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rsi
  int v8; // eax
  const __m128i *v9; // rdx
  __m128i v10; // xmm2
  __int64 v11; // rax
  char v12; // cl
  __int64 v13; // rax
  int v14; // eax
  unsigned int v15; // ebx
  const __m128i *v17; // rdx
  __int64 v18; // rax
  __m128i v19; // xmm0
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  __int64 v22; // r10
  __int64 v23; // rbx
  unsigned __int8 *v24; // rax
  __int64 v25; // rdi
  __int32 v26; // edx
  unsigned __int8 *v27; // rax
  __int64 v28; // rdi
  __int32 v29; // edx
  unsigned __int8 *v30; // rax
  __int64 v31; // rdi
  __int32 v32; // edx
  unsigned __int8 *v33; // rax
  __int64 v34; // rdi
  __int32 v35; // edx
  unsigned __int8 *v36; // rax
  __int64 v37; // rdi
  __int32 v38; // edx
  unsigned __int8 *v39; // rax
  __m128i v40; // xmm4
  __int64 v41; // rax
  __int64 v42; // r9
  __m128i v43; // xmm5
  __m128i v44; // xmm6
  __int32 v45; // edx
  unsigned __int64 v46; // rdx
  __m128i *v47; // rax
  int v48; // edx
  unsigned int v49; // eax
  __m128i *v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  int v55; // esi
  __int64 v56; // rax
  _QWORD *v57; // rdi
  __int64 v58; // r13
  __int64 v59; // rcx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // [rsp+0h] [rbp-230h]
  __int128 v63; // [rsp+0h] [rbp-230h]
  __int64 v64; // [rsp+8h] [rbp-228h]
  __int64 v65; // [rsp+10h] [rbp-220h]
  unsigned __int64 v66; // [rsp+18h] [rbp-218h]
  __int64 v67; // [rsp+20h] [rbp-210h]
  unsigned int v68; // [rsp+2Ch] [rbp-204h]
  unsigned int v69; // [rsp+30h] [rbp-200h]
  __int64 v70; // [rsp+30h] [rbp-200h]
  unsigned __int64 v71; // [rsp+40h] [rbp-1F0h]
  unsigned int v72; // [rsp+48h] [rbp-1E8h]
  __int16 v73; // [rsp+4Eh] [rbp-1E2h]
  __m128i *v74; // [rsp+60h] [rbp-1D0h] BYREF
  int v75; // [rsp+68h] [rbp-1C8h]
  __m128i v76; // [rsp+70h] [rbp-1C0h] BYREF
  __m128i v77; // [rsp+80h] [rbp-1B0h] BYREF
  __m128i v78; // [rsp+90h] [rbp-1A0h] BYREF
  __m128i v79; // [rsp+A0h] [rbp-190h] BYREF
  __m128i v80; // [rsp+B0h] [rbp-180h] BYREF
  __m128i v81; // [rsp+C0h] [rbp-170h] BYREF
  __m128i v82; // [rsp+D0h] [rbp-160h] BYREF
  __m128i v83; // [rsp+E0h] [rbp-150h] BYREF
  __m128i v84; // [rsp+F0h] [rbp-140h] BYREF
  __m128i v85; // [rsp+100h] [rbp-130h] BYREF
  __m128i v86; // [rsp+110h] [rbp-120h] BYREF
  __m128i v87; // [rsp+120h] [rbp-110h] BYREF
  __m128i *v88; // [rsp+130h] [rbp-100h] BYREF
  __int64 v89; // [rsp+138h] [rbp-F8h]
  __m128i v90; // [rsp+140h] [rbp-F0h] BYREF
  __m128i v91; // [rsp+150h] [rbp-E0h]
  __m128i v92; // [rsp+160h] [rbp-D0h]
  __m128i v93; // [rsp+170h] [rbp-C0h]
  __m128i v94; // [rsp+180h] [rbp-B0h]
  __m128i v95; // [rsp+190h] [rbp-A0h]
  __m128i v96; // [rsp+1A0h] [rbp-90h]
  __m128i v97; // [rsp+1B0h] [rbp-80h]

  v4 = *(unsigned __int16 *)(a2 + 96);
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 48LL);
  v6 = *(_QWORD *)(v5 + 8);
  v73 = *(_WORD *)v5;
  v72 = sub_36D7800(*(_QWORD *)(a2 + 112));
  if ( v72 == 4 )
    sub_C64ED0("Cannot store to pointer that points to constant memory space", 1u);
  v7 = *(_QWORD *)(a2 + 80);
  v74 = (__m128i *)v7;
  if ( v7 )
  {
    sub_B96E90((__int64)&v74, v7, 1);
    v8 = *(_DWORD *)(a2 + 72);
    v9 = *(const __m128i **)(a2 + 40);
    v75 = v8;
    v10 = _mm_loadu_si128(v9);
    v88 = v74;
    v76 = v10;
    if ( v74 )
    {
      sub_B96E90((__int64)&v88, (__int64)v74, 1);
      v8 = v75;
    }
  }
  else
  {
    v8 = *(_DWORD *)(a2 + 72);
    v17 = *(const __m128i **)(a2 + 40);
    v75 = v8;
    v88 = 0;
    v76 = _mm_loadu_si128(v17);
  }
  LODWORD(v89) = v8;
  v71 = sub_36E1BC0(a1, (__int64)&v88, (__int64)&v76, a2);
  if ( v88 )
    sub_B91220((__int64)&v88, (__int64)v88);
  if ( (unsigned __int16)(v4 - 17) <= 0xD3u )
    LOWORD(v4) = word_4456580[v4 - 1];
  if ( (unsigned __int16)v4 <= 1u || (unsigned __int16)(v4 - 504) <= 7u )
    BUG();
  v11 = 16LL * ((unsigned __int16)v4 - 1);
  v12 = byte_444C4A0[v11 + 8];
  v13 = *(_QWORD *)&byte_444C4A0[v11];
  LOBYTE(v89) = v12;
  v88 = (__m128i *)v13;
  v69 = sub_CA1930(&v88);
  v68 = sub_36D79E0((unsigned __int16)v4, 0);
  v88 = &v90;
  v89 = 0xC00000000LL;
  v14 = *(_DWORD *)(a2 + 24);
  if ( v14 == 554 )
  {
    v65 = 4;
    v53 = *(_QWORD *)(a2 + 40);
    v79 = _mm_loadu_si128((const __m128i *)(v53 + 40));
    v80 = _mm_loadu_si128((const __m128i *)(v53 + 80));
    v81 = _mm_loadu_si128((const __m128i *)(v53 + 120));
    v19 = _mm_loadu_si128((const __m128i *)(v53 + 160));
    LODWORD(v89) = 4;
    v82 = v19;
    v90 = v79;
    v91 = v80;
    v92 = v81;
    v93 = v19;
    v20 = *(_QWORD *)(v53 + 200);
    v21 = *(unsigned int *)(v53 + 208);
  }
  else if ( v14 == 555 )
  {
    v51 = *(_QWORD *)(a1 + 1136);
    v15 = 0;
    if ( *(_DWORD *)(v51 + 344) <= 0x63u || *(_DWORD *)(v51 + 336) <= 0x57u )
      goto LABEL_14;
    v65 = 8;
    v52 = *(_QWORD *)(a2 + 40);
    v79 = _mm_loadu_si128((const __m128i *)(v52 + 40));
    v80 = _mm_loadu_si128((const __m128i *)(v52 + 80));
    v81 = _mm_loadu_si128((const __m128i *)(v52 + 120));
    v82 = _mm_loadu_si128((const __m128i *)(v52 + 160));
    v83 = _mm_loadu_si128((const __m128i *)(v52 + 200));
    v84 = _mm_loadu_si128((const __m128i *)(v52 + 240));
    v85 = _mm_loadu_si128((const __m128i *)(v52 + 280));
    v19 = _mm_loadu_si128((const __m128i *)(v52 + 320));
    LODWORD(v89) = 8;
    v86 = v19;
    v90 = v79;
    v91 = v80;
    v92 = v81;
    v93 = v82;
    v94 = v83;
    v95 = v84;
    v96 = v85;
    v97 = v19;
    v20 = *(_QWORD *)(v52 + 360);
    v21 = *(unsigned int *)(v52 + 368);
  }
  else
  {
    v15 = 0;
    if ( v14 != 553 )
      goto LABEL_14;
    v65 = 2;
    v18 = *(_QWORD *)(a2 + 40);
    v79 = _mm_loadu_si128((const __m128i *)(v18 + 40));
    v19 = _mm_loadu_si128((const __m128i *)(v18 + 80));
    LODWORD(v89) = 2;
    v80 = v19;
    v90 = v79;
    v91 = v19;
    v20 = *(_QWORD *)(v18 + 120);
    v21 = *(unsigned int *)(v18 + 128);
  }
  v66 = v21;
  v67 = v20;
  if ( (unsigned __int8)sub_307AB50(v73, v6, v21) || v73 == 37 )
  {
    v22 = 3;
    v23 = 32;
    v73 = 7;
  }
  else
  {
    v23 = v69;
    v22 = v68;
  }
  v70 = v22;
  v77.m128i_i64[0] = 0;
  v77.m128i_i32[2] = 0;
  v78.m128i_i64[0] = 0;
  v78.m128i_i32[2] = 0;
  sub_36DF750(a1, v67, v66, (__int64)&v78, (__int64)&v77, v19);
  v24 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v71, (__int64)&v74, 7, 0, 1u, v19, 0);
  v25 = *(_QWORD *)(a1 + 64);
  v79.m128i_i32[2] = v26;
  v79.m128i_i64[0] = (__int64)v24;
  v27 = sub_3400BD0(v25, HIDWORD(v71), (__int64)&v74, 7, 0, 1u, v19, 0);
  v28 = *(_QWORD *)(a1 + 64);
  v80.m128i_i32[2] = v29;
  v80.m128i_i64[0] = (__int64)v27;
  v30 = sub_3400BD0(v28, v72, (__int64)&v74, 7, 0, 1u, v19, 0);
  v31 = *(_QWORD *)(a1 + 64);
  v81.m128i_i32[2] = v32;
  v81.m128i_i64[0] = (__int64)v30;
  v33 = sub_3400BD0(v31, v65, (__int64)&v74, 7, 0, 1u, v19, 0);
  v34 = *(_QWORD *)(a1 + 64);
  v82.m128i_i32[2] = v35;
  v82.m128i_i64[0] = (__int64)v33;
  v36 = sub_3400BD0(v34, v70, (__int64)&v74, 7, 0, 1u, v19, 0);
  v37 = *(_QWORD *)(a1 + 64);
  v83.m128i_i32[2] = v38;
  v83.m128i_i64[0] = (__int64)v36;
  v39 = sub_3400BD0(v37, v23, (__int64)&v74, 7, 0, 1u, v19, 0);
  v40 = _mm_loadu_si128(&v78);
  v84.m128i_i64[0] = (__int64)v39;
  v41 = (unsigned int)v89;
  v85 = v40;
  v42 = v64;
  v43 = _mm_loadu_si128(&v77);
  v44 = _mm_loadu_si128(&v76);
  v84.m128i_i32[2] = v45;
  v46 = (unsigned int)v89 + 9LL;
  v86 = v43;
  v87 = v44;
  if ( v46 > HIDWORD(v89) )
  {
    sub_C8D5F0((__int64)&v88, &v90, v46, 0x10u, v62, v64);
    v41 = (unsigned int)v89;
  }
  v47 = &v88[v41];
  *v47 = _mm_loadu_si128(&v79);
  v47[1] = _mm_loadu_si128(&v80);
  v47[2] = _mm_loadu_si128(&v81);
  v47[3] = _mm_loadu_si128(&v82);
  v47[4] = _mm_loadu_si128(&v83);
  v47[5] = _mm_loadu_si128(&v84);
  v47[6] = _mm_loadu_si128(&v85);
  v47[7] = _mm_loadu_si128(&v86);
  v47[8] = _mm_loadu_si128(&v87);
  v48 = *(_DWORD *)(a2 + 24);
  v49 = v89 + 9;
  LODWORD(v89) = v89 + 9;
  if ( v48 == 554 )
  {
    v79.m128i_i64[0] = 0x100000E75LL;
    v54 = sub_36D6650(v73, 3710, 3703, 3705, 0x100000E7CLL, 3698, 0x100000E75LL);
    goto LABEL_36;
  }
  if ( v48 != 555 )
  {
    if ( v48 != 553 )
    {
LABEL_27:
      v50 = v88;
      v15 = 0;
      goto LABEL_28;
    }
    v79.m128i_i64[0] = 0x100000E74LL;
    v54 = sub_36D6650(v73, 3709, 3702, 3704, 0x100000E7BLL, 3697, 0x100000E74LL);
LABEL_36:
    v55 = v54;
    v15 = HIDWORD(v54);
    if ( !BYTE4(v54) )
    {
      v50 = v88;
      goto LABEL_28;
    }
    goto LABEL_42;
  }
  if ( v73 == 7 )
  {
    v55 = 3706;
    goto LABEL_43;
  }
  if ( v73 != 12 )
    goto LABEL_27;
  v55 = 3699;
LABEL_42:
  v49 = v89;
LABEL_43:
  *((_QWORD *)&v63 + 1) = v49;
  v15 = 1;
  *(_QWORD *)&v63 = v88;
  v56 = sub_33F7800(*(_QWORD **)(a1 + 64), v55, (__int64)&v74, 1u, 0, v42, v63);
  v57 = *(_QWORD **)(a1 + 64);
  v58 = v56;
  v79.m128i_i64[0] = *(_QWORD *)(a2 + 112);
  sub_33E4DA0(v57, v56, v79.m128i_i64, 1);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v58, v59, v60, v61);
  sub_3421DB0(v58);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  v50 = v88;
LABEL_28:
  if ( v50 != &v90 )
    _libc_free((unsigned __int64)v50);
LABEL_14:
  if ( v74 )
    sub_B91220((__int64)&v74, (__int64)v74);
  return v15;
}
