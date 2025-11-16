// Function: sub_26431D0
// Address: 0x26431d0
//
_QWORD *__fastcall sub_26431D0(unsigned __int64 *a1, __int64 *a2, char *a3, __int64 a4)
{
  unsigned __int64 v4; // r8
  __int64 v5; // r14
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rsi
  __int64 v14; // r11
  __int64 v15; // r10
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // r13
  __int64 v19; // r15
  __int64 v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r11
  __int64 v25; // r9
  __int64 *v26; // r10
  __int64 v27; // rdi
  __int64 v28; // rcx
  __m128i v29; // xmm4
  __m128i v30; // xmm5
  _QWORD *result; // rax
  __int64 v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rcx
  __int64 v36; // r15
  __int64 v37; // r8
  _QWORD *v38; // r12
  __m128i v39; // kr00_16
  __int64 v40; // rsi
  __int64 v41; // rcx
  __int64 v42; // rdx
  __m128i v43; // xmm5
  unsigned __int128 v44; // kr10_16
  _QWORD *v45; // rbx
  __int64 v46; // rdx
  unsigned __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // rax
  __m128i v52; // xmm1
  __int64 v53; // r9
  char *v54; // r15
  __int64 v55; // r8
  __int64 v56; // rax
  __m128i v57; // xmm6
  __m128i v58; // xmm7
  __int64 v59; // rsi
  __int64 v60; // rcx
  __int64 v61; // rdx
  __int64 v62; // r10
  char *v63; // r14
  __m128i v64; // xmm6
  __m128i v65; // xmm7
  __int64 v66; // rax
  __int64 v67; // rax
  __m128i v68; // xmm3
  __int64 v69; // [rsp+8h] [rbp-148h]
  __int64 v70; // [rsp+10h] [rbp-140h]
  __int64 v71; // [rsp+18h] [rbp-138h]
  __m128i v72; // [rsp+18h] [rbp-138h]
  __int64 v73; // [rsp+20h] [rbp-130h]
  __int64 v74; // [rsp+20h] [rbp-130h]
  __int64 *v75; // [rsp+28h] [rbp-128h]
  __int64 v76; // [rsp+28h] [rbp-128h]
  __m128i v77; // [rsp+28h] [rbp-128h]
  __int64 v78; // [rsp+30h] [rbp-120h]
  __m128i v79; // [rsp+30h] [rbp-120h]
  __int64 v80; // [rsp+30h] [rbp-120h]
  __int64 v81; // [rsp+38h] [rbp-118h]
  __int64 v82; // [rsp+40h] [rbp-110h]
  __m128i v83; // [rsp+40h] [rbp-110h]
  char *v84; // [rsp+40h] [rbp-110h]
  __int64 v85; // [rsp+48h] [rbp-108h]
  __m128i v88; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v89; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v90; // [rsp+80h] [rbp-D0h] BYREF
  __m128i v91; // [rsp+90h] [rbp-C0h]
  __m128i v92; // [rsp+A0h] [rbp-B0h] BYREF
  __m128i v93; // [rsp+B0h] [rbp-A0h]
  __m128i v94; // [rsp+C0h] [rbp-90h] BYREF
  __m128i v95; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v96; // [rsp+E0h] [rbp-70h] BYREF
  __m128i v97; // [rsp+F0h] [rbp-60h]
  __m128i v98; // [rsp+100h] [rbp-50h] BYREF
  __m128i v99; // [rsp+110h] [rbp-40h]

  v4 = a4 - (_QWORD)a3;
  v5 = (a4 - (__int64)a3) >> 3;
  v7 = *a2;
  v8 = a1[2];
  if ( *a2 == v8 )
  {
    v66 = a1[3];
    if ( v8 - v66 < v4 )
    {
      sub_26406F0(a1, v5 - ((v8 - v66) >> 3));
      v8 = a1[2];
      v66 = a1[3];
    }
    v94.m128i_i64[1] = v66;
    v67 = a1[4];
    v94.m128i_i64[0] = v8;
    v95.m128i_i64[0] = v67;
    v95.m128i_i64[1] = a1[5];
    sub_263F500(v94.m128i_i64, -v5);
    v96 = v94;
    v97 = v95;
    result = sub_2642610(&v98, a3, a4, (char **)&v96);
    v68 = _mm_loadu_si128(&v95);
    *((__m128i *)a1 + 1) = _mm_loadu_si128(&v94);
    *((__m128i *)a1 + 2) = v68;
  }
  else
  {
    v9 = a1[6];
    if ( v7 == v9 )
    {
      v46 = a1[8];
      v47 = ((v46 - v9) >> 3) - 1;
      if ( v5 > v47 )
      {
        sub_26407C0(a1, v5 - v47);
        v9 = a1[6];
        v46 = a1[8];
      }
      v48 = a1[7];
      v94.m128i_i64[0] = v9;
      v95.m128i_i64[0] = v46;
      v94.m128i_i64[1] = v48;
      v95.m128i_i64[1] = a1[9];
      sub_263F500(v94.m128i_i64, v5);
      v49 = a1[8];
      v50 = a1[7];
      v51 = a1[9];
      v96.m128i_i64[0] = a1[6];
      v97.m128i_i64[0] = v49;
      v96.m128i_i64[1] = v50;
      v97.m128i_i64[1] = v51;
      result = sub_2642610(&v98, a3, a4, (char **)&v96);
      v52 = _mm_loadu_si128(&v95);
      *((__m128i *)a1 + 3) = _mm_loadu_si128(&v94);
      *((__m128i *)a1 + 4) = v52;
    }
    else
    {
      v10 = a1[5];
      v11 = a2[3];
      v12 = v7 - a2[1];
      v13 = a1[9];
      v14 = a1[4];
      v15 = a1[7];
      v16 = ((((v11 - v10) >> 3) - 1) << 6) + (v12 >> 3);
      v17 = (v14 - v8) >> 3;
      v18 = ((((v13 - v10) >> 3) - 1) << 6) + ((v9 - v15) >> 3);
      v19 = v16 + v17;
      if ( (unsigned __int64)(v18 + v17) >> 1 <= v16 + v17 )
      {
        v32 = a1[8];
        v33 = ((v32 - v9) >> 3) - 1;
        if ( v5 > v33 )
        {
          sub_26407C0(a1, v5 - v33);
          v9 = a1[6];
          v15 = a1[7];
          v32 = a1[8];
          v13 = a1[9];
        }
        v88.m128i_i64[0] = v9;
        v89.m128i_i64[1] = v13;
        v89.m128i_i64[0] = v32;
        v88.m128i_i64[1] = v15;
        sub_263F500(v88.m128i_i64, v5);
        v34 = a1[7];
        v35 = a1[9];
        v36 = a1[6];
        v76 = a1[8];
        v99.m128i_i64[0] = v76;
        v98.m128i_i64[0] = v36;
        v80 = v34;
        v98.m128i_i64[1] = v34;
        v74 = v35;
        v99.m128i_i64[1] = v35;
        sub_263F500(v98.m128i_i64, v16 - v18);
        v38 = (_QWORD *)v99.m128i_i64[1];
        v39 = v98;
        v84 = (char *)v99.m128i_i64[0];
        if ( v5 >= v37 )
        {
          v53 = a1[7];
          v54 = &a3[8 * v37];
          v55 = a1[8];
          v56 = a1[9];
          v96.m128i_i64[0] = a1[6];
          v72.m128i_i64[0] = v96.m128i_i64[0];
          v96.m128i_i64[1] = v53;
          v72.m128i_i64[1] = v53;
          v97.m128i_i64[0] = v55;
          v77.m128i_i64[0] = v55;
          v97.m128i_i64[1] = v56;
          v77.m128i_i64[1] = v56;
          sub_2642610(&v98, v54, a4, (char **)&v96);
          v94 = v98;
          v98 = v39;
          v95 = v99;
          v96 = v72;
          v99.m128i_i64[0] = (__int64)v84;
          v97 = v77;
          v99.m128i_i64[1] = (__int64)v38;
          sub_26428A0(&v92, v98.m128i_i64, v96.m128i_i64, &v94);
          v97.m128i_i64[1] = (__int64)v38;
          v57 = _mm_loadu_si128(&v88);
          v58 = _mm_loadu_si128(&v89);
          v96 = v39;
          v97.m128i_i64[0] = (__int64)v84;
          *((__m128i *)a1 + 3) = v57;
          *((__m128i *)a1 + 4) = v58;
          return sub_2642610(&v98, a3, (__int64)v54, (char **)&v96);
        }
        else
        {
          v90 = *((__m128i *)a1 + 3);
          v91 = *((__m128i *)a1 + 4);
          sub_263F500(v90.m128i_i64, -v5);
          v40 = a1[6];
          v41 = a1[7];
          v42 = a1[8];
          v97.m128i_i64[1] = a1[9];
          v95.m128i_i64[1] = v97.m128i_i64[1];
          v96.m128i_i64[0] = v40;
          v92 = v90;
          v96.m128i_i64[1] = v41;
          v97.m128i_i64[0] = v42;
          v93 = v91;
          v94.m128i_i64[0] = v40;
          v94.m128i_i64[1] = v41;
          v95.m128i_i64[0] = v42;
          sub_26428A0(&v98, v92.m128i_i64, v94.m128i_i64, &v96);
          v43 = _mm_loadu_si128(&v89);
          v44 = (unsigned __int128)v90;
          *((__m128i *)a1 + 3) = _mm_loadu_si128(&v88);
          *((__m128i *)a1 + 4) = v43;
          v45 = (_QWORD *)v91.m128i_i64[1];
          if ( (_QWORD *)v91.m128i_i64[1] == v38 )
          {
            v98.m128i_i64[0] = v36;
            v98.m128i_i64[1] = v80;
            v99.m128i_i64[0] = v76;
            v99.m128i_i64[1] = v74;
            sub_2642980(v96.m128i_i64, v39.m128i_i64[0], (char *)v44, v98.m128i_i64);
          }
          else
          {
            v96.m128i_i64[0] = v36;
            v96.m128i_i64[1] = v80;
            v97.m128i_i64[0] = v76;
            v97.m128i_i64[1] = v74;
            while ( 1 )
            {
              --v45;
              sub_2642980(v98.m128i_i64, *((__int64 *)&v44 + 1), (char *)v44, v96.m128i_i64);
              if ( v45 == v38 )
                break;
              v97 = v99;
              v96 = v98;
              v44 = __PAIR128__(*v45, *v45 + 512LL);
            }
            sub_2642980(v96.m128i_i64, v39.m128i_i64[0], v84, v98.m128i_i64);
          }
          v97.m128i_i64[1] = (__int64)v38;
          v96 = v39;
          v97.m128i_i64[0] = (__int64)v84;
          return sub_2642610(&v98, a3, a4, (char **)&v96);
        }
      }
      else
      {
        v20 = a1[3];
        if ( v8 - v20 < v4 )
        {
          sub_26406F0(a1, v5 - ((v8 - v20) >> 3));
          v8 = a1[2];
          v20 = a1[3];
          v14 = a1[4];
          v10 = a1[5];
        }
        v88.m128i_i64[0] = v8;
        v88.m128i_i64[1] = v20;
        v89.m128i_i64[0] = v14;
        v89.m128i_i64[1] = v10;
        sub_263F500(v88.m128i_i64, -v5);
        v21 = a1[5];
        v22 = a1[2];
        v23 = a1[4];
        v82 = a1[3];
        v98.m128i_i64[1] = v82;
        v78 = v21;
        v99.m128i_i64[1] = v21;
        v85 = v22;
        v98.m128i_i64[0] = v22;
        v81 = v23;
        v99.m128i_i64[0] = v23;
        sub_263F500(v98.m128i_i64, v19);
        v24 = v98.m128i_i64[1];
        v25 = v99.m128i_i64[0];
        if ( v5 > v19 )
        {
          v59 = a1[2];
          v60 = a1[3];
          v61 = a1[4];
          v96.m128i_i64[0] = v98.m128i_i64[0];
          v62 = a1[5];
          v98.m128i_i64[0] = v59;
          v94 = v88;
          v98.m128i_i64[1] = v60;
          v99.m128i_i64[0] = v61;
          v63 = &a3[8 * (v5 - v19)];
          v95 = v89;
          v96.m128i_i64[1] = v24;
          v97.m128i_i64[0] = v25;
          v97.m128i_i64[1] = v99.m128i_i64[1];
          v99.m128i_i64[1] = v62;
          sub_26428A0(&v92, v98.m128i_i64, v96.m128i_i64, &v94);
          v96 = v92;
          v97 = v93;
          sub_2642610(&v98, a3, (__int64)v63, (char **)&v96);
          v64 = _mm_loadu_si128(&v88);
          v65 = _mm_loadu_si128(&v89);
          v96.m128i_i64[0] = v85;
          *((__m128i *)a1 + 1) = v64;
          v96.m128i_i64[1] = v82;
          *((__m128i *)a1 + 2) = v65;
          v97.m128i_i64[0] = v81;
          v97.m128i_i64[1] = v78;
          return sub_2642610(&v98, v63, a4, (char **)&v96);
        }
        else
        {
          v69 = v99.m128i_i64[1];
          v70 = v99.m128i_i64[0];
          v90 = *((__m128i *)a1 + 1);
          v71 = v98.m128i_i64[1];
          v73 = v98.m128i_i64[0];
          v91 = *((__m128i *)a1 + 2);
          sub_263F500(v90.m128i_i64, v5);
          v75 = v26;
          v96 = v88;
          v97 = v89;
          v94 = v90;
          v95 = v91;
          v92 = *((__m128i *)a1 + 1);
          v93 = *((__m128i *)a1 + 2);
          sub_26428A0(&v98, v26, v94.m128i_i64, &v96);
          v94 = v90;
          v27 = v82;
          v98.m128i_i64[0] = v85;
          v28 = v81;
          v96.m128i_i64[1] = v71;
          v29 = _mm_loadu_si128(&v88);
          v95 = v91;
          v30 = _mm_loadu_si128(&v89);
          v96.m128i_i64[0] = v73;
          v79.m128i_i64[0] = v73;
          v79.m128i_i64[1] = v71;
          v97.m128i_i64[0] = v70;
          v83.m128i_i64[0] = v70;
          v97.m128i_i64[1] = v69;
          v83.m128i_i64[1] = v69;
          v98.m128i_i64[1] = v27;
          v99.m128i_i64[0] = v28;
          v99.m128i_i64[1] = v21;
          *((__m128i *)a1 + 1) = v29;
          *((__m128i *)a1 + 2) = v30;
          sub_2642740(v75, (__int64)&v94, v96.m128i_i64, (__int64)&v98);
          v94 = v79;
          v95 = v83;
          sub_263F500(v94.m128i_i64, -v5);
          v96 = v94;
          v97 = v95;
          return sub_2642610(&v98, a3, a4, (char **)&v96);
        }
      }
    }
  }
  return result;
}
