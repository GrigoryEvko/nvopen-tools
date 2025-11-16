// Function: sub_37A9DB0
// Address: 0x37a9db0
//
__m128i *__fastcall sub_37A9DB0(__int64 a1, __int64 a2, int a3)
{
  _QWORD *v4; // rax
  bool v5; // zf
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // r14
  __int64 v9; // rcx
  unsigned __int64 v10; // r13
  __int64 v11; // r15
  _QWORD *v12; // rax
  __int64 v13; // rax
  __int64 v14; // r8
  unsigned __int16 v15; // cx
  __int64 v16; // r14
  unsigned int v17; // edx
  unsigned __int64 v18; // r15
  __int64 v19; // rdx
  __m128i v20; // xmm0
  __m128i v21; // xmm2
  __int64 v22; // rax
  __m128i v23; // xmm1
  __int64 v24; // rsi
  unsigned __int16 v25; // ax
  _QWORD *v26; // r15
  __int64 v27; // rdi
  __m128i *v28; // rax
  __int32 v29; // edx
  __m128i *v30; // r12
  __m128i v32; // xmm3
  __int64 v33; // rax
  unsigned int v34; // edx
  __int64 v35; // rax
  __int64 v36; // r8
  unsigned int v37; // edx
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  int v41; // r9d
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // r8
  int v45; // r9d
  unsigned int v46; // edx
  unsigned __int16 v47; // cx
  __int64 v48; // rdx
  __int64 *v49; // rax
  unsigned __int16 v50; // ax
  __int64 v51; // r9
  unsigned int v52; // r11d
  __int64 v53; // rdx
  unsigned __int64 v54; // rsi
  bool v55; // al
  unsigned __int16 v56; // ax
  unsigned __int64 v57; // rdx
  __int64 v58; // [rsp+10h] [rbp-150h]
  __int64 v59; // [rsp+18h] [rbp-148h]
  __int64 v60; // [rsp+18h] [rbp-148h]
  unsigned __int16 v61; // [rsp+20h] [rbp-140h]
  int v62; // [rsp+20h] [rbp-140h]
  __int64 *v63; // [rsp+20h] [rbp-140h]
  __int64 v64; // [rsp+28h] [rbp-138h]
  __int64 v65; // [rsp+28h] [rbp-138h]
  char v66; // [rsp+28h] [rbp-138h]
  __int64 v67; // [rsp+28h] [rbp-138h]
  __int64 v68; // [rsp+30h] [rbp-130h]
  int v69; // [rsp+38h] [rbp-128h]
  unsigned __int16 v70; // [rsp+38h] [rbp-128h]
  unsigned __int64 v71; // [rsp+40h] [rbp-120h]
  unsigned int v72; // [rsp+40h] [rbp-120h]
  char v73; // [rsp+40h] [rbp-120h]
  unsigned int v74; // [rsp+40h] [rbp-120h]
  __int64 v75; // [rsp+40h] [rbp-120h]
  __int64 v76; // [rsp+48h] [rbp-118h]
  __int64 v77; // [rsp+48h] [rbp-118h]
  unsigned __int16 v78; // [rsp+48h] [rbp-118h]
  unsigned int v79; // [rsp+54h] [rbp-10Ch]
  __int16 v80; // [rsp+54h] [rbp-10Ch]
  unsigned __int64 v81; // [rsp+58h] [rbp-108h]
  const __m128i *v82; // [rsp+58h] [rbp-108h]
  __int64 v83; // [rsp+B0h] [rbp-B0h] BYREF
  int v84; // [rsp+B8h] [rbp-A8h]
  __m128i v85; // [rsp+C0h] [rbp-A0h] BYREF
  unsigned __int64 v86; // [rsp+D0h] [rbp-90h]
  unsigned __int64 v87; // [rsp+D8h] [rbp-88h]
  __m128i v88; // [rsp+E0h] [rbp-80h]
  __int64 v89; // [rsp+F0h] [rbp-70h]
  unsigned __int64 v90; // [rsp+F8h] [rbp-68h]
  __int64 v91; // [rsp+100h] [rbp-60h]
  int v92; // [rsp+108h] [rbp-58h]
  unsigned __int64 v93; // [rsp+110h] [rbp-50h]
  unsigned __int64 v94; // [rsp+118h] [rbp-48h]
  __m128i v95; // [rsp+120h] [rbp-40h]

  v4 = *(_QWORD **)(a2 + 40);
  v5 = *(_DWORD *)(a2 + 24) == 470;
  v6 = v4[5];
  v76 = v4[6];
  v81 = v6;
  v79 = *((_DWORD *)v4 + 12);
  if ( v5 )
  {
    v7 = v4[20];
    v10 = v4[21];
    v9 = 15;
    v8 = v4[10];
    v11 = v4[11];
  }
  else
  {
    v7 = v4[25];
    v8 = v4[15];
    v9 = 20;
    v10 = v4[26];
    v11 = v4[16];
  }
  v12 = &v4[v9];
  v69 = *((_DWORD *)v12 + 2);
  v68 = *v12;
  v71 = *(_QWORD *)(a2 + 104);
  if ( a3 == 1 )
  {
    v65 = a2;
    v33 = sub_379AB60(a1, v6, v76);
    v79 = v34;
    v72 = v34;
    v81 = v33;
    v35 = sub_379AB60(a1, v8, v11);
    v36 = v65;
    v16 = v35;
    v38 = *(_QWORD *)(v81 + 48) + 16LL * v72;
    v18 = v37 | v11 & 0xFFFFFFFF00000000LL;
    v39 = *(_QWORD *)(v38 + 8);
    v85.m128i_i16[0] = *(_WORD *)v38;
    v85.m128i_i64[1] = v39;
    if ( v85.m128i_i16[0] )
    {
      v73 = (unsigned __int16)(v85.m128i_i16[0] - 176) <= 0x34u;
      v66 = v73;
      v41 = word_4456340[v85.m128i_u16[0] - 1];
    }
    else
    {
      v40 = sub_3007240((__int64)&v85);
      v36 = v65;
      v41 = v40;
      v66 = BYTE4(v40);
      v73 = BYTE4(v40);
    }
    v42 = v7;
    v59 = v36;
    v62 = v41;
    v43 = sub_379AB60(a1, v7, v10);
    v44 = v59;
    v45 = v62;
    v7 = v43;
    v10 = v10 & 0xFFFFFFFF00000000LL | v46;
    v48 = *(_QWORD *)(v59 + 104);
    v85.m128i_i16[0] = *(_WORD *)(v59 + 96);
    v47 = v85.m128i_i16[0];
    v85.m128i_i64[1] = v48;
    if ( v85.m128i_i16[0] )
    {
      if ( (unsigned __int16)(v85.m128i_i16[0] - 17) <= 0xD3u )
      {
        v47 = word_4456580[v85.m128i_u16[0] - 1];
        v48 = 0;
      }
    }
    else
    {
      v58 = v48;
      v55 = sub_30070B0((__int64)&v85);
      v47 = 0;
      v48 = v58;
      v45 = v62;
      v44 = v59;
      if ( v55 )
      {
        v56 = sub_3009970((__int64)&v85, v42, v58, 0, v59);
        v44 = v59;
        v45 = v62;
        v47 = v56;
      }
    }
    v5 = v73 == 0;
    v49 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
    v60 = v44;
    LODWORD(v83) = v45;
    v63 = v49;
    v74 = v47;
    BYTE4(v83) = v66;
    v67 = v48;
    if ( v5 )
    {
      v50 = sub_2D43050(v47, v45);
      v14 = v60;
      v53 = v67;
      v52 = v74;
    }
    else
    {
      v50 = sub_2D43AD0(v47, v45);
      v52 = v74;
      v53 = v67;
      v14 = v60;
    }
    v54 = 0;
    if ( !v50 )
    {
      v75 = v14;
      v50 = sub_3009450(v63, v52, v53, v83, v14, v51);
      v14 = v75;
      v54 = v57;
    }
    v71 = v54;
    v15 = v50;
  }
  else
  {
    v61 = *(_WORD *)(a2 + 96);
    v64 = a2;
    if ( a3 != 3 )
      BUG();
    v13 = sub_379AB60(a1, v8, v11);
    v14 = v64;
    v15 = v61;
    v16 = v13;
    v18 = v17 | v11 & 0xFFFFFFFF00000000LL;
  }
  v19 = *(_QWORD *)(v14 + 40);
  v20 = _mm_loadu_si128((const __m128i *)v19);
  v5 = *(_DWORD *)(v14 + 24) == 470;
  v86 = v81;
  v87 = v79 | v76 & 0xFFFFFFFF00000000LL;
  v85 = v20;
  if ( v5 )
  {
    v32 = _mm_loadu_si128((const __m128i *)(v19 + 40));
    v89 = v16;
    v90 = v18;
    v91 = v68;
    v93 = v7;
    v92 = v69;
    v22 = 200;
    v94 = v10;
    v88 = v32;
  }
  else
  {
    v21 = _mm_loadu_si128((const __m128i *)(v19 + 80));
    v89 = v16;
    v90 = v18;
    v91 = v68;
    v93 = v7;
    v94 = v10;
    v92 = v69;
    v22 = 240;
    v88 = v21;
  }
  v23 = _mm_loadu_si128((const __m128i *)(v19 + v22));
  v24 = *(_QWORD *)(v14 + 80);
  v25 = *(_WORD *)(v14 + 32);
  v26 = *(_QWORD **)(a1 + 8);
  v83 = v24;
  v95 = v23;
  v80 = (v25 >> 7) & 7;
  v82 = *(const __m128i **)(v14 + 112);
  if ( v24 )
  {
    v70 = v15;
    v77 = v14;
    sub_B96E90((__int64)&v83, v24, 1);
    v27 = *(_QWORD *)(a1 + 8);
    v14 = v77;
    v15 = v70;
  }
  else
  {
    v27 = (__int64)v26;
  }
  v78 = v15;
  v84 = *(_DWORD *)(v14 + 72);
  v28 = sub_33ED250(v27, 1, 0);
  v30 = sub_33E6FD0(v26, (unsigned __int64)v28, v29, v78, v71, (__int64)&v83, (unsigned __int64 *)&v85, 7, v82, v80);
  if ( v83 )
    sub_B91220((__int64)&v83, v83);
  return v30;
}
