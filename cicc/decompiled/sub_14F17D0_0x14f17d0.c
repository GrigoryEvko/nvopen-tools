// Function: sub_14F17D0
// Address: 0x14f17d0
//
__int64 *__fastcall sub_14F17D0(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 *v3; // r13
  __int64 v4; // rcx
  const __m128i *v5; // r12
  const __m128i *v6; // rdx
  const __m128i *v7; // rcx
  const __m128i *v8; // r14
  const __m128i *v9; // rbx
  __int64 v10; // rcx
  const __m128i *v11; // r15
  const __m128i *v12; // rcx
  __int64 v13; // rcx
  const __m128i *v14; // rcx
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 v17; // rcx
  __int64 v18; // rcx
  const __m128i *v19; // r13
  const __m128i *v20; // rbx
  __int64 v22; // rdi
  const __m128i *v23; // r14
  __int64 v24; // rsi
  __int64 v25; // rsi
  const __m128i *v26; // rbx
  const __m128i *v27; // r10
  __int64 v28; // rsi
  const __m128i *v29; // r15
  __int64 v30; // rcx
  __int64 v31; // rcx
  _QWORD *v32; // rdx
  __int64 v33; // rdi
  unsigned __int64 v34; // rsi
  __int64 v35; // rdi
  const __m128i *v36; // r8
  const __m128i *v37; // r12
  _QWORD *v38; // rbx
  const __m128i *v39; // r13
  __int64 v40; // rdi
  const __m128i *v41; // r14
  __int64 v42; // rsi
  __int64 v43; // rsi
  const __m128i *v44; // rdx
  const __m128i *v45; // rbx
  _QWORD *v46; // r12
  __int64 v47; // rdi
  unsigned int v48; // esi
  __int64 v49; // rsi
  __int64 v50; // rdx
  _QWORD *v51; // rbx
  __int64 v52; // rdi
  unsigned int v53; // esi
  __int64 v54; // rcx
  __int64 v55; // rsi
  __m128i *v56; // rsi
  const __m128i *v57; // rbx
  __m128i *v59; // rsi
  const char *v60; // rcx
  __m128i *v61; // rsi
  __m128i *v62; // rsi
  __m128i *v63; // rsi
  const __m128i *v64; // [rsp+0h] [rbp-C0h]
  __int64 v65; // [rsp+8h] [rbp-B8h]
  const __m128i *v66; // [rsp+10h] [rbp-B0h]
  const __m128i *v67; // [rsp+18h] [rbp-A8h]
  const __m128i **v68; // [rsp+20h] [rbp-A0h]
  const __m128i **v69; // [rsp+20h] [rbp-A0h]
  const __m128i *v70; // [rsp+20h] [rbp-A0h]
  _QWORD *v71; // [rsp+20h] [rbp-A0h]
  __int64 v72; // [rsp+28h] [rbp-98h]
  __int64 v73; // [rsp+30h] [rbp-90h]
  __int64 v74; // [rsp+38h] [rbp-88h]
  __int64 v75; // [rsp+40h] [rbp-80h]
  __int64 v76; // [rsp+48h] [rbp-78h]
  __int64 *v78; // [rsp+50h] [rbp-70h]
  const __m128i **v79; // [rsp+50h] [rbp-70h]
  __int64 v80; // [rsp+50h] [rbp-70h]
  const __m128i *v81; // [rsp+50h] [rbp-70h]
  __int64 v82; // [rsp+50h] [rbp-70h]
  const __m128i *v83; // [rsp+58h] [rbp-68h]
  const __m128i **v84; // [rsp+58h] [rbp-68h]
  const __m128i *v85; // [rsp+58h] [rbp-68h]
  const __m128i *v86; // [rsp+58h] [rbp-68h]
  const __m128i **v87; // [rsp+58h] [rbp-68h]
  const __m128i *v88; // [rsp+60h] [rbp-60h]
  const __m128i *v89; // [rsp+60h] [rbp-60h]
  const __m128i *v90; // [rsp+60h] [rbp-60h]
  const __m128i *v91; // [rsp+60h] [rbp-60h]
  __int64 v92; // [rsp+68h] [rbp-58h]
  __int64 v93[2]; // [rsp+70h] [rbp-50h] BYREF
  char v94; // [rsp+80h] [rbp-40h]
  char v95; // [rsp+81h] [rbp-3Fh]

  v2 = a2;
  v3 = a1;
  v68 = (const __m128i **)(a2 + 147);
  v4 = a2[149];
  v5 = (const __m128i *)a2[147];
  v6 = (const __m128i *)a2[148];
  a2[147] = 0;
  v73 = v4;
  v7 = (const __m128i *)a2[151];
  v8 = (const __m128i *)a2[150];
  v9 = (const __m128i *)a2[153];
  a2[148] = 0;
  v67 = v7;
  v10 = a2[152];
  a2[149] = 0;
  v11 = (const __m128i *)a2[156];
  v72 = v10;
  v12 = (const __m128i *)a2[154];
  a2[150] = 0;
  v66 = v12;
  v13 = a2[155];
  a2[151] = 0;
  a2[152] = 0;
  v74 = v13;
  a2[153] = 0;
  a2[154] = 0;
  a2[155] = 0;
  v14 = (const __m128i *)a2[157];
  a2[156] = 0;
  v64 = v14;
  v15 = a2[158];
  a2[157] = 0;
  v75 = v15;
  v16 = a2[159];
  a2[158] = 0;
  v92 = v16;
  v17 = a2[160];
  a2[159] = 0;
  v65 = v17;
  v18 = a2[161];
  a2[160] = 0;
  v76 = v18;
  a2[161] = 0;
  if ( v6 != v5 )
  {
    v83 = v8;
    v19 = v6 - 1;
    v88 = v9;
    v20 = v5;
    do
    {
      v22 = a2[69];
      v23 = v19;
      v24 = v19->m128i_u32[2];
      if ( (unsigned int)v24 >= -1431655765 * (unsigned int)((a2[70] - v22) >> 3) )
      {
        v56 = (__m128i *)a2[148];
        if ( v56 == (__m128i *)a2[149] )
        {
          sub_14F1350(v68, v56, v19);
        }
        else
        {
          if ( v56 )
          {
            *v56 = _mm_loadu_si128(v19);
            v56 = (__m128i *)a2[148];
          }
          a2[148] = v56 + 1;
        }
      }
      else
      {
        v25 = *(_QWORD *)(v22 + 24 * v24 + 16);
        if ( !v25 || *(_BYTE *)(v25 + 16) > 0x10u )
        {
          v2 = a2;
          v95 = 1;
          v5 = v20;
          v8 = v83;
          v3 = a1;
          v57 = v88;
          v93[0] = (__int64)"Expected a constant";
          goto LABEL_51;
        }
        sub_15E5440(v19->m128i_i64[0], v25);
      }
      --v19;
    }
    while ( v20 != v23 );
    v2 = a2;
    v8 = v83;
    v5 = v20;
    v3 = a1;
    v9 = v88;
  }
  v84 = (const __m128i **)(v2 + 150);
  if ( v67 != v8 )
  {
    v89 = v9;
    v26 = v67 - 1;
    v27 = v11;
    do
    {
      v28 = v2[69];
      v29 = v26;
      v30 = v26->m128i_u32[2];
      if ( (unsigned int)v30 >= -1431655765 * (unsigned int)((v2[70] - v28) >> 3) )
      {
        v59 = (__m128i *)v2[151];
        if ( v59 == (__m128i *)v2[152] )
        {
          v71 = v2;
          v81 = v27;
          sub_14F14D0(v84, v59, v26);
          v27 = v81;
          v2 = v71;
        }
        else
        {
          if ( v59 )
          {
            *v59 = _mm_loadu_si128(v26);
            v59 = (__m128i *)v2[151];
          }
          v2[151] = v59 + 1;
        }
      }
      else
      {
        v31 = *(_QWORD *)(v28 + 24 * v30 + 16);
        if ( !v31 || *(_BYTE *)(v31 + 16) > 0x10u )
        {
          v11 = v27;
LABEL_69:
          v57 = v89;
          goto LABEL_70;
        }
        v32 = (_QWORD *)v26->m128i_i64[0];
        if ( *(_BYTE *)(v26->m128i_i64[0] + 16) == 1 && *v32 != *(_QWORD *)v31 )
        {
          v95 = 1;
          v57 = v89;
          v11 = v27;
          v60 = "Alias and aliasee types don't match";
          goto LABEL_71;
        }
        if ( *(v32 - 3) )
        {
          v33 = *(v32 - 2);
          v34 = *(v32 - 1) & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v34 = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = *(_QWORD *)(v33 + 16) & 3LL | v34;
        }
        *(v32 - 3) = v31;
        v35 = *(_QWORD *)(v31 + 8);
        *(v32 - 2) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 16) = *(_QWORD *)(v35 + 16) & 3LL | (unsigned __int64)(v32 - 2);
        *(v32 - 1) = *(v32 - 1) & 3LL | (v31 + 8);
        *(_QWORD *)(v31 + 8) = v32 - 3;
      }
      --v26;
    }
    while ( v8 != v29 );
    v9 = v89;
    v11 = v27;
  }
  v69 = (const __m128i **)(v2 + 153);
  v36 = v66 - 1;
  if ( v66 != v9 )
  {
    v90 = v8;
    v85 = v5;
    v37 = v9;
    v38 = v2;
    v78 = v3;
    v39 = v66 - 1;
    do
    {
      v40 = v38[69];
      v41 = v39;
      v42 = v39->m128i_u32[2];
      if ( (unsigned int)v42 >= -1431655765 * (unsigned int)((v38[70] - v40) >> 3) )
      {
        v61 = (__m128i *)v38[154];
        if ( v61 == (__m128i *)v38[155] )
        {
          sub_14F1650(v69, v61, v39);
        }
        else
        {
          if ( v61 )
          {
            *v61 = _mm_loadu_si128(v39);
            v61 = (__m128i *)v38[154];
          }
          v38[154] = v61 + 1;
        }
      }
      else
      {
        v43 = *(_QWORD *)(v40 + 24 * v42 + 16);
        if ( !v43 || *(_BYTE *)(v43 + 16) > 0x10u )
        {
          v2 = v38;
          v8 = v90;
          v57 = v37;
          v3 = v78;
          v5 = v85;
          goto LABEL_70;
        }
        sub_15E3F20(v39->m128i_i64[0]);
      }
      --v39;
    }
    while ( v37 != v41 );
    v2 = v38;
    v8 = v90;
    v9 = v37;
    v3 = v78;
    v5 = v85;
  }
  v79 = (const __m128i **)(v2 + 156);
  v44 = v64 - 1;
  if ( v11 == v64 )
  {
LABEL_40:
    v50 = v65;
    v89 = v9;
    v51 = v2;
    v87 = (const __m128i **)(v2 + 159);
    while ( 1 )
    {
      while ( 1 )
      {
        if ( v92 == v50 )
        {
          *v3 = 1;
          v57 = v89;
          v93[0] = 0;
          sub_14ECA90(v93);
          goto LABEL_52;
        }
        v52 = v51[69];
        v53 = *(_DWORD *)(v50 - 8);
        if ( v53 < -1431655765 * (unsigned int)((v51[70] - v52) >> 3) )
          break;
        v63 = (__m128i *)v51[160];
        if ( v63 == (__m128i *)v51[161] )
        {
          v82 = v50 - 16;
          sub_14F1650(v87, v63, (const __m128i *)(v50 - 16));
          v50 = v82;
        }
        else
        {
          if ( v63 )
            *v63 = _mm_loadu_si128((const __m128i *)(v50 - 16));
          v51[160] += 16LL;
          v50 -= 16;
        }
      }
      v54 = 24LL * v53;
      v55 = *(_QWORD *)(v52 + v54 + 16);
      if ( !v55 || *(_BYTE *)(v55 + 16) > 0x10u )
        break;
      v80 = v50;
      sub_15E3D80(*(_QWORD *)(v50 - 16), v55, v50, v54, v36);
      v50 = v80 - 16;
    }
    v2 = v51;
    goto LABEL_69;
  }
  v91 = v9;
  v45 = v64 - 1;
  v86 = v5;
  v46 = v2;
  while ( 1 )
  {
    v47 = v46[69];
    v48 = v45->m128i_u32[2];
    v70 = v45;
    if ( v48 >= -1431655765 * (unsigned int)((v46[70] - v47) >> 3) )
    {
      v62 = (__m128i *)v46[157];
      if ( v62 == (__m128i *)v46[158] )
      {
        sub_14F1650(v79, v62, v45);
      }
      else
      {
        if ( v62 )
          *v62 = _mm_loadu_si128(v45);
        v46[157] += 16LL;
      }
      goto LABEL_38;
    }
    v49 = *(_QWORD *)(v47 + 24LL * v48 + 16);
    if ( !v49 || *(_BYTE *)(v49 + 16) > 0x10u )
      break;
    sub_15E40D0(v45->m128i_i64[0], v49, v44);
LABEL_38:
    --v45;
    if ( v11 == v70 )
    {
      v2 = v46;
      v9 = v91;
      v5 = v86;
      goto LABEL_40;
    }
  }
  v2 = v46;
  v57 = v91;
  v5 = v86;
LABEL_70:
  v95 = 1;
  v60 = "Expected a constant";
LABEL_71:
  v93[0] = (__int64)v60;
LABEL_51:
  v94 = 3;
  sub_14EE4B0(v3, (__int64)(v2 + 1), (__int64)v93);
LABEL_52:
  if ( v92 )
    j_j___libc_free_0(v92, v76 - v92);
  if ( v11 )
    j_j___libc_free_0(v11, v75 - (_QWORD)v11);
  if ( v57 )
    j_j___libc_free_0(v57, v74 - (_QWORD)v57);
  if ( v8 )
    j_j___libc_free_0(v8, v72 - (_QWORD)v8);
  if ( v5 )
    j_j___libc_free_0(v5, v73 - (_QWORD)v5);
  return v3;
}
