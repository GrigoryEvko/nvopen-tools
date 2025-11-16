// Function: sub_3775FE0
// Address: 0x3775fe0
//
unsigned __int8 *__fastcall sub_3775FE0(
        _QWORD *a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        unsigned int a5,
        int a6,
        __m128i a7)
{
  unsigned int v7; // r13d
  __int64 v8; // rbx
  __int64 v9; // rax
  unsigned int *v10; // rax
  __int64 v11; // r12
  __int64 v12; // rsi
  __int64 v14; // rax
  __int16 v15; // dx
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  unsigned __int64 v20; // rax
  unsigned __int32 v21; // r12d
  unsigned __int64 v22; // r15
  int v23; // eax
  int v24; // r9d
  __int64 v25; // rdx
  unsigned __int8 *v26; // rax
  int v27; // r9d
  __int64 v28; // r12
  unsigned __int64 v29; // rdx
  unsigned __int64 v30; // r13
  __int64 v31; // r15
  unsigned __int64 v32; // rax
  __int128 v33; // rax
  __int64 v34; // r9
  unsigned int v35; // edx
  __int64 v36; // rax
  unsigned __int16 v37; // dx
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned __int64 v41; // rax
  __m128i v42; // kr00_16
  __int64 v43; // rax
  int v44; // r9d
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned int v47; // edx
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // rax
  __int64 v51; // rdx
  unsigned __int64 v52; // rax
  unsigned __int8 *v53; // r12
  __int128 v55; // [rsp-30h] [rbp-180h]
  __int128 v56; // [rsp-20h] [rbp-170h]
  __int64 v57; // [rsp-10h] [rbp-160h]
  unsigned __int64 v58; // [rsp+8h] [rbp-148h]
  unsigned __int64 v59; // [rsp+10h] [rbp-140h]
  __int64 v60; // [rsp+20h] [rbp-130h]
  unsigned int v61; // [rsp+28h] [rbp-128h]
  __int64 *v62; // [rsp+28h] [rbp-128h]
  __int64 v63; // [rsp+30h] [rbp-120h]
  __int64 v64; // [rsp+30h] [rbp-120h]
  __int64 v65; // [rsp+30h] [rbp-120h]
  __int64 *v66; // [rsp+38h] [rbp-118h]
  unsigned int v67; // [rsp+38h] [rbp-118h]
  unsigned __int64 v68; // [rsp+38h] [rbp-118h]
  unsigned int v72; // [rsp+4Ch] [rbp-104h]
  __int64 v73; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v74; // [rsp+78h] [rbp-D8h]
  __int64 v75; // [rsp+80h] [rbp-D0h] BYREF
  int v76; // [rsp+88h] [rbp-C8h]
  __m128i v77; // [rsp+90h] [rbp-C0h] BYREF
  __m128i v78; // [rsp+A0h] [rbp-B0h] BYREF
  __int64 v79; // [rsp+B0h] [rbp-A0h] BYREF
  char v80; // [rsp+B8h] [rbp-98h]
  __int64 v81; // [rsp+C0h] [rbp-90h] BYREF
  char v82; // [rsp+C8h] [rbp-88h]
  __int64 v83; // [rsp+D0h] [rbp-80h]
  __int64 v84; // [rsp+D8h] [rbp-78h]
  __int64 v85; // [rsp+E0h] [rbp-70h]
  __int64 v86; // [rsp+E8h] [rbp-68h]
  __int64 v87; // [rsp+F0h] [rbp-60h]
  __int64 v88; // [rsp+F8h] [rbp-58h]
  __int64 v89; // [rsp+100h] [rbp-50h]
  __int64 v90; // [rsp+108h] [rbp-48h]
  __int64 v91; // [rsp+110h] [rbp-40h] BYREF
  __int64 v92; // [rsp+118h] [rbp-38h]

  v8 = 16LL * a5;
  v9 = *a4;
  v73 = a2;
  v74 = a3;
  v10 = (unsigned int *)(v8 + v9);
  v11 = *(_QWORD *)v10;
  v12 = *(_QWORD *)(*(_QWORD *)v10 + 80LL);
  v75 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v75, v12, 1);
    v10 = (unsigned int *)(v8 + *a4);
  }
  v76 = *(_DWORD *)(v11 + 72);
  v14 = *(_QWORD *)(*(_QWORD *)v10 + 48LL) + 16LL * v10[2];
  v15 = *(_WORD *)v14;
  v77.m128i_i64[1] = *(_QWORD *)(v14 + 8);
  v77.m128i_i16[0] = v15;
  if ( (_WORD)v73 )
  {
    if ( (_WORD)v73 == 1 || (unsigned __int16)(v73 - 504) <= 7u )
      goto LABEL_43;
    v17 = 16LL * ((unsigned __int16)v73 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v17];
    LOBYTE(v17) = byte_444C4A0[v17 + 8];
  }
  else
  {
    v16 = sub_3007260((__int64)&v73);
    v85 = v16;
    v86 = v17;
  }
  v91 = v16;
  LOBYTE(v92) = v17;
  v59 = (unsigned int)sub_CA1930(&v91);
  if ( !v77.m128i_i16[0] )
  {
    v18 = sub_3007260((__int64)&v77);
    v83 = v18;
    v84 = v19;
    goto LABEL_7;
  }
  if ( v77.m128i_i16[0] == 1 || (unsigned __int16)(v77.m128i_i16[0] - 504) <= 7u )
LABEL_43:
    BUG();
  v19 = 16LL * (v77.m128i_u16[0] - 1);
  v18 = *(_QWORD *)&byte_444C4A0[v19];
  LOBYTE(v19) = byte_444C4A0[v19 + 8];
LABEL_7:
  LOBYTE(v92) = v19;
  v91 = v18;
  v20 = sub_CA1930(&v91);
  v21 = v77.m128i_i32[0];
  v22 = v59 / v20;
  v66 = (__int64 *)a1[8];
  v63 = v77.m128i_i64[1];
  LOWORD(v23) = sub_2D43050(v77.m128i_i16[0], v59 / v20);
  v25 = 0;
  if ( !(_WORD)v23 )
  {
    v23 = sub_3009400(v66, v21, v63, (unsigned int)v22, 0);
    HIWORD(v7) = HIWORD(v23);
  }
  LOWORD(v7) = v23;
  v64 = v25;
  v61 = v7;
  v57 = *(_QWORD *)(*a4 + v8);
  v26 = sub_33FAF80((__int64)a1, 167, (__int64)&v75, v7, v25, v24, a7);
  v27 = v57;
  v67 = 1;
  v28 = (__int64)v26;
  v30 = v29;
  v72 = a5 + 1;
  if ( v72 != a6 )
  {
    v31 = v64;
    do
    {
      v65 = 16LL * v72;
      v36 = *(_QWORD *)(*(_QWORD *)(*a4 + v65) + 48LL) + 16LL * *(unsigned int *)(*a4 + v65 + 8);
      v37 = *(_WORD *)v36;
      v38 = *(_QWORD *)(v36 + 8);
      v78.m128i_i16[0] = v37;
      v78.m128i_i64[1] = v38;
      if ( v77.m128i_i16[0] == v37 )
      {
        if ( v77.m128i_i64[1] == v38 || v37 )
        {
          LODWORD(v32) = v67;
          goto LABEL_14;
        }
      }
      else if ( v37 )
      {
        if ( v37 == 1 || (unsigned __int16)(v37 - 504) <= 7u )
          goto LABEL_43;
        v40 = 16LL * (v37 - 1);
        v39 = *(_QWORD *)&byte_444C4A0[v40];
        LOBYTE(v40) = byte_444C4A0[v40 + 8];
        goto LABEL_18;
      }
      v39 = sub_3007260((__int64)&v78);
      v91 = v39;
      v92 = v40;
LABEL_18:
      v81 = v39;
      v82 = v40;
      v41 = sub_CA1930(&v81);
      v42 = v78;
      v62 = (__int64 *)a1[8];
      v58 = v59 / v41;
      LOWORD(v43) = sub_2D43050(v78.m128i_i16[0], v59 / v41);
      v45 = 0;
      if ( !(_WORD)v43 )
      {
        v43 = sub_3009400(v62, v42.m128i_u32[0], v42.m128i_i64[1], (unsigned int)v58, 0);
        v60 = v43;
      }
      v46 = v60;
      v31 = v45;
      LOWORD(v46) = v43;
      v60 = v46;
      v61 = v46;
      v28 = (__int64)sub_33FAF80((__int64)a1, 234, (__int64)&v75, (unsigned int)v46, v45, v44, a7);
      v30 = v47 | v30 & 0xFFFFFFFF00000000LL;
      if ( v77.m128i_i16[0] )
      {
        if ( v77.m128i_i16[0] == 1 || (unsigned __int16)(v77.m128i_i16[0] - 504) <= 7u )
          goto LABEL_43;
        v49 = *(_QWORD *)&byte_444C4A0[16 * v77.m128i_u16[0] - 16];
        LOBYTE(v48) = byte_444C4A0[16 * v77.m128i_u16[0] - 8];
      }
      else
      {
        v89 = sub_3007260((__int64)&v77);
        v49 = v89;
        v90 = v48;
      }
      v80 = v48;
      v79 = v49 * v67;
      v68 = sub_CA1930(&v79);
      if ( v78.m128i_i16[0] )
      {
        if ( v78.m128i_i16[0] == 1 || (unsigned __int16)(v78.m128i_i16[0] - 504) <= 7u )
          goto LABEL_43;
        v51 = 16LL * (v78.m128i_u16[0] - 1);
        v50 = *(_QWORD *)&byte_444C4A0[v51];
        LOBYTE(v51) = byte_444C4A0[v51 + 8];
      }
      else
      {
        v50 = sub_3007260((__int64)&v78);
        v87 = v50;
        v88 = v51;
      }
      v82 = v51;
      v81 = v50;
      v52 = sub_CA1930(&v81);
      a7 = _mm_loadu_si128(&v78);
      v77 = a7;
      v32 = v68 / v52;
LABEL_14:
      v67 = v32 + 1;
      *(_QWORD *)&v33 = sub_3400EE0((__int64)a1, (unsigned int)v32, (__int64)&v75, 0, a7);
      v56 = *(_OWORD *)(*a4 + 16LL * v72);
      *((_QWORD *)&v55 + 1) = v30;
      *(_QWORD *)&v55 = v28;
      ++v72;
      v28 = sub_340F900(a1, 0x9Du, (__int64)&v75, v61, v31, v34, v55, v56, v33);
      v30 = v35 | v30 & 0xFFFFFFFF00000000LL;
    }
    while ( a6 != v72 );
  }
  v53 = sub_33FAF80((__int64)a1, 234, (__int64)&v75, (unsigned int)v73, v74, v27, a7);
  if ( v75 )
    sub_B91220((__int64)&v75, v75);
  return v53;
}
