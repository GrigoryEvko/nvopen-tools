// Function: sub_37A3390
// Address: 0x37a3390
//
__m128i *__fastcall sub_37A3390(__int64 *a1, unsigned __int64 a2)
{
  __int64 v4; // r9
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v6; // rax
  unsigned __int16 v7; // si
  __int64 v8; // r8
  __int64 v9; // rax
  __int64 v10; // r10
  unsigned int v11; // eax
  __int32 v12; // edx
  _QWORD *v13; // rbx
  __int64 v14; // rcx
  _QWORD *v15; // rcx
  __int64 v16; // rdi
  int v17; // r14d
  __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rsi
  __m128i v21; // rax
  unsigned int v22; // ecx
  __int64 v23; // rbx
  __int64 v24; // rax
  bool v25; // zf
  __int64 *v26; // rax
  unsigned __int16 v27; // ax
  __int64 v28; // r9
  unsigned __int64 v29; // r14
  _QWORD *v30; // r8
  unsigned __int16 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rdx
  __m128i v34; // xmm2
  __m128i v35; // xmm3
  __int64 v36; // rax
  __m128i v37; // xmm1
  __int64 *v38; // rdi
  const __m128i *v39; // r9
  unsigned __int16 v40; // bx
  unsigned __int64 v41; // rax
  __int32 v42; // edx
  __m128i *v43; // r13
  unsigned __int64 v45; // rdx
  __int64 v46; // rax
  bool v47; // al
  __int64 v48; // rdx
  __int64 v49; // r8
  __int64 v50; // rdx
  __m128i v51; // xmm4
  __m128i v52; // xmm5
  __int64 v53; // rdx
  __int64 *v54; // [rsp+0h] [rbp-140h]
  unsigned int v55; // [rsp+0h] [rbp-140h]
  __int64 v56; // [rsp+10h] [rbp-130h]
  unsigned __int64 v57; // [rsp+18h] [rbp-128h]
  __m128i v58; // [rsp+20h] [rbp-120h] BYREF
  char v59; // [rsp+37h] [rbp-109h]
  __int64 v60; // [rsp+38h] [rbp-108h]
  __m128i *v61; // [rsp+40h] [rbp-100h]
  __int64 v62; // [rsp+48h] [rbp-F8h]
  const __m128i *v63; // [rsp+50h] [rbp-F0h]
  unsigned __int64 v64; // [rsp+58h] [rbp-E8h]
  __int64 v65; // [rsp+60h] [rbp-E0h]
  unsigned __int64 v66; // [rsp+68h] [rbp-D8h]
  __int64 v67; // [rsp+70h] [rbp-D0h]
  __int64 v68; // [rsp+78h] [rbp-C8h]
  __int64 v69; // [rsp+80h] [rbp-C0h]
  __int64 v70; // [rsp+88h] [rbp-B8h]
  unsigned int v71; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v72; // [rsp+98h] [rbp-A8h]
  __int64 v73; // [rsp+A0h] [rbp-A0h] BYREF
  int v74; // [rsp+A8h] [rbp-98h]
  __m128i v75; // [rsp+B0h] [rbp-90h] BYREF
  __m128i v76; // [rsp+C0h] [rbp-80h]
  __m128i v77; // [rsp+D0h] [rbp-70h]
  __int64 v78; // [rsp+E0h] [rbp-60h]
  int v79; // [rsp+E8h] [rbp-58h]
  __int64 v80; // [rsp+F0h] [rbp-50h]
  unsigned __int64 v81; // [rsp+F8h] [rbp-48h]
  __m128i v82; // [rsp+100h] [rbp-40h]

  v4 = *a1;
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v6 = *(__int16 **)(a2 + 48);
  v7 = *v6;
  v8 = *((_QWORD *)v6 + 1);
  v9 = a1[1];
  v10 = *(_QWORD *)(v9 + 64);
  if ( v5 == sub_2D56A50 )
  {
    v61 = &v75;
    sub_2FE6CC0((__int64)&v75, v4, v10, v7, v8);
    LOWORD(v11) = v75.m128i_i16[4];
    LOWORD(v71) = v75.m128i_i16[4];
    v72 = v76.m128i_i64[0];
  }
  else
  {
    v11 = v5(*a1, *(_QWORD *)(v9 + 64), v7, v8);
    v61 = &v75;
    v71 = v11;
    v72 = v53;
  }
  v12 = *(_DWORD *)(a2 + 24);
  v13 = *(_QWORD **)(a2 + 40);
  if ( v12 == 470 )
  {
    v56 = v13[21];
    v14 = 15;
    v57 = v13[20];
  }
  else
  {
    v57 = v13[25];
    v56 = v13[26];
    v14 = 20;
  }
  v15 = &v13[v14];
  v16 = *v15;
  LODWORD(v15) = *((_DWORD *)v15 + 2);
  v60 = v16;
  LODWORD(v63) = (_DWORD)v15;
  if ( (_WORD)v11 )
  {
    LOBYTE(v64) = (unsigned __int16)(v11 - 176) <= 0x34u;
    v59 = v64;
    v17 = word_4456340[(unsigned __int16)v11 - 1];
  }
  else
  {
    v58.m128i_i32[0] = v12;
    v46 = sub_3007240((__int64)&v71);
    v12 = v58.m128i_i32[0];
    v17 = v46;
    v69 = v46;
    v59 = BYTE4(v46);
    LOBYTE(v64) = BYTE4(v46);
  }
  v18 = *(_QWORD *)(a2 + 80);
  v73 = v18;
  if ( v18 )
  {
    sub_B96E90((__int64)&v73, v18, 1);
    v12 = *(_DWORD *)(a2 + 24);
    v13 = *(_QWORD **)(a2 + 40);
  }
  v74 = *(_DWORD *)(a2 + 72);
  v19 = 10;
  if ( v12 != 470 )
    v19 = 15;
  v20 = v13[v19];
  v21.m128i_i64[0] = sub_379AB60((__int64)a1, v20, v13[v19 + 1]);
  v22 = *(unsigned __int16 *)(a2 + 96);
  v23 = *(_QWORD *)(a2 + 104);
  v58 = v21;
  v75.m128i_i16[0] = v22;
  v75.m128i_i64[1] = v23;
  if ( (_WORD)v22 )
  {
    if ( (unsigned __int16)(v22 - 17) <= 0xD3u )
    {
      v23 = 0;
      LOWORD(v22) = word_4456580[v22 - 1];
    }
  }
  else
  {
    v55 = v22;
    v47 = sub_30070B0((__int64)v61);
    LOWORD(v22) = v55;
    if ( v47 )
    {
      LOWORD(v22) = sub_3009970((__int64)v61, v20, v48, v55, v49);
      v23 = v50;
    }
  }
  v24 = a1[1];
  v25 = (_BYTE)v64 == 0;
  LODWORD(v70) = v17;
  v26 = *(__int64 **)(v24 + 64);
  v64 = (unsigned __int16)v22;
  v54 = v26;
  BYTE4(v70) = v59;
  if ( v25 )
  {
    v27 = sub_2D43050(v22, v17);
    v29 = 0;
    v30 = (_QWORD *)v64;
    if ( v27 )
      goto LABEL_16;
  }
  else
  {
    v27 = sub_2D43AD0(v22, v17);
    v29 = 0;
    v30 = (_QWORD *)v64;
    if ( v27 )
      goto LABEL_16;
  }
  v27 = sub_3009450(v54, (unsigned int)v30, v23, v70, (__int64)v30, v28);
  v29 = v45;
LABEL_16:
  v31 = v27;
  v67 = sub_379AB60((__int64)a1, v57, v56);
  v68 = v32;
  v65 = v67;
  v66 = v56 & 0xFFFFFFFF00000000LL | (unsigned int)v32;
  v33 = *(_QWORD *)(a2 + 40);
  v25 = *(_DWORD *)(a2 + 24) == 470;
  v75 = _mm_loadu_si128((const __m128i *)v33);
  if ( v25 )
  {
    v51 = _mm_loadu_si128((const __m128i *)(v33 + 40));
    v81 = v66;
    v36 = 200;
    v52 = _mm_load_si128(&v58);
    v80 = v67;
    v79 = (int)v63;
    v78 = v60;
    v76 = v51;
    v77 = v52;
  }
  else
  {
    v34 = _mm_loadu_si128((const __m128i *)(v33 + 80));
    v80 = v67;
    v35 = _mm_load_si128(&v58);
    v81 = v66;
    v36 = 240;
    v78 = v60;
    v76 = v34;
    v79 = (int)v63;
    v77 = v35;
  }
  v37 = _mm_loadu_si128((const __m128i *)(v33 + v36));
  v38 = (__int64 *)a1[1];
  v39 = *(const __m128i **)(a2 + 112);
  v40 = *(_WORD *)(a2 + 32);
  v62 = 6;
  v82 = v37;
  v63 = v39;
  v64 = (unsigned __int64)v38;
  v41 = sub_33E5110(v38, v71, v72, 1, 0);
  v43 = sub_33E79D0((_QWORD *)v64, v41, v42, v31, v29, (__int64)&v73, (unsigned __int64 *)v61, v62, v63, (v40 >> 7) & 7);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v43, 1);
  if ( v73 )
    sub_B91220((__int64)&v73, v73);
  return v43;
}
