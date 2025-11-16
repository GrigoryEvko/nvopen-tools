// Function: sub_3878580
// Address: 0x3878580
//
__int64 ***__fastcall sub_3878580(
        __int64 *a1,
        __int64 a2,
        __m128 a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rbx
  __int64 v15; // r14
  __int64 v16; // rax
  int v17; // r9d
  __int64 v18; // r8
  __int64 v19; // r15
  __int64 v20; // rax
  __m128i *v21; // rax
  unsigned int v22; // eax
  __int64 v23; // rdx
  __m128i *v24; // r12
  __int64 *m128i_i64; // r14
  __int64 v26; // r15
  __int64 ***v27; // r14
  int v28; // r8d
  int v29; // r9d
  double v30; // xmm4_8
  double v31; // xmm5_8
  unsigned __int64 v32; // r12
  __m128i *v33; // rbx
  __int64 ***v34; // r15
  __int64 v35; // r14
  __int64 v36; // r11
  __m128i *v37; // r15
  __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // rsi
  __int64 v42; // rdi
  __int64 v43; // rdi
  __int64 v44; // rax
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r10
  __int64 *v48; // r15
  __int64 v49; // rax
  __int64 v50; // r15
  __int64 *v51; // rcx
  __int64 v52; // rax
  __int64 v53; // r11
  __int64 ***v54; // r15
  __int64 **v55; // rsi
  __int64 **v56; // r8
  __int64 ***v57; // r9
  __int64 v58; // rcx
  __int64 **v59; // rdx
  __int64 v61; // rax
  __int64 v62; // rsi
  double v63; // xmm4_8
  double v64; // xmm5_8
  __int64 v65; // rax
  double v66; // xmm4_8
  double v67; // xmm5_8
  __int64 ***v68; // r14
  double v69; // xmm4_8
  double v70; // xmm5_8
  __int64 ***v71; // rax
  __int64 v72; // r8
  int v73; // r9d
  __int64 v74; // rax
  __int64 ***v75; // r14
  double v76; // xmm4_8
  double v77; // xmm5_8
  __int64 ***v78; // rax
  __int64 v79; // r8
  int v80; // r9d
  __int64 v81; // rdx
  __int64 ***v82; // [rsp+10h] [rbp-110h]
  __int64 v83; // [rsp+10h] [rbp-110h]
  __int64 v84; // [rsp+18h] [rbp-108h]
  __int64 v85; // [rsp+18h] [rbp-108h]
  __int64 **v86; // [rsp+20h] [rbp-100h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  __int64 v88; // [rsp+28h] [rbp-F8h]
  __int64 **v89; // [rsp+30h] [rbp-F0h] BYREF
  __int64 v90; // [rsp+38h] [rbp-E8h]
  __int64 *v91[4]; // [rsp+40h] [rbp-E0h] BYREF
  __m128i *v92; // [rsp+60h] [rbp-C0h] BYREF
  __int64 v93; // [rsp+68h] [rbp-B8h]
  _BYTE v94[176]; // [rsp+70h] [rbp-B0h] BYREF

  v11 = *a1;
  v12 = sub_1456040(*(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL * ((unsigned int)*(_QWORD *)(a2 + 40) - 1)));
  v13 = sub_1456E10(v11, v12);
  v14 = *(_QWORD *)(a2 + 32);
  v86 = (__int64 **)v13;
  v92 = (__m128i *)v94;
  v93 = 0x800000000LL;
  v15 = v14 + 8LL * *(_QWORD *)(a2 + 40);
  if ( v14 == v15 )
  {
    v24 = (__m128i *)v94;
    v23 = 0;
    m128i_i64 = (__int64 *)v94;
  }
  else
  {
    do
    {
      v16 = sub_3873B70((__int64)a1, *(_QWORD *)(v15 - 8));
      v18 = *(_QWORD *)(v15 - 8);
      v19 = v16;
      v20 = (unsigned int)v93;
      if ( (unsigned int)v93 >= HIDWORD(v93) )
      {
        v88 = *(_QWORD *)(v15 - 8);
        sub_16CD150((__int64)&v92, v94, 0, 16, v18, v17);
        v20 = (unsigned int)v93;
        v18 = v88;
      }
      v21 = &v92[v20];
      v15 -= 8;
      v21->m128i_i64[0] = v19;
      v21->m128i_i64[1] = v18;
      v22 = v93 + 1;
      LODWORD(v93) = v93 + 1;
    }
    while ( v14 != v15 );
    v23 = v22;
    v24 = v92;
    m128i_i64 = v92[v22].m128i_i64;
  }
  v26 = *(_QWORD *)(*a1 + 56);
  sub_3872D20((__int64 *)&v89, v24, v23);
  if ( v91[0] )
    sub_3873790(v24->m128i_i64, m128i_i64, v91[0], v90, v26);
  else
    sub_386FCA0(v24->m128i_i64, m128i_i64, v26);
  v27 = 0;
  j_j___libc_free_0((unsigned __int64)v91[0]);
  v32 = (unsigned __int64)v92;
  v33 = &v92[(unsigned int)v93];
  if ( v33 != v92 )
  {
    v34 = 0;
    while ( 1 )
    {
      v35 = *(_QWORD *)v32;
      if ( !v34 )
      {
        v62 = *(_QWORD *)(v32 + 8);
        v32 += 16LL;
        v34 = (__int64 ***)sub_3875200(a1, v62, *(double *)a3.m128_u64, *(double *)a4.m128i_i64, a5);
        goto LABEL_33;
      }
      v36 = (__int64)*v34;
      if ( *((_BYTE *)*v34 + 8) != 15 )
      {
        v42 = *(_QWORD *)(v32 + 8);
        v87 = v42;
        v32 += 16LL;
        v85 = sub_1456040(v42);
        if ( *(_BYTE *)(v85 + 8) != 15 )
        {
          if ( sub_1456260(v42) )
          {
            v65 = sub_1480620(*a1, v42, 0);
            v68 = sub_38761C0(a1, v65, v86, a3, *(double *)a4.m128i_i64, a5, a6, v66, v67, a9, a10);
            v71 = sub_38744E0(a1, (__int64)v34, v86, a3, *(double *)a4.m128i_i64, a5, a6, v69, v70, a9, a10);
            v34 = (__int64 ***)sub_3874770(
                                 (__int64)a1,
                                 0xDu,
                                 (__int64)v71,
                                 (__int64)v68,
                                 v72,
                                 v73,
                                 *(double *)a3.m128_u64,
                                 *(double *)a4.m128i_i64,
                                 a5);
          }
          else
          {
            v75 = sub_38761C0(a1, v42, v86, a3, *(double *)a4.m128i_i64, a5, a6, v63, v64, a9, a10);
            v78 = sub_38744E0(a1, (__int64)v34, v86, a3, *(double *)a4.m128i_i64, a5, a6, v76, v77, a9, a10);
            v81 = (__int64)v78;
            if ( *((_BYTE *)v78 + 16) <= 0x10u )
            {
              v81 = (__int64)v75;
              v75 = v78;
            }
            v34 = (__int64 ***)sub_3874770(
                                 (__int64)a1,
                                 0xBu,
                                 v81,
                                 (__int64)v75,
                                 v79,
                                 v80,
                                 *(double *)a3.m128_u64,
                                 *(double *)a4.m128i_i64,
                                 a5);
          }
          goto LABEL_33;
        }
        v43 = *a1;
        v89 = v91;
        v90 = 0x400000000LL;
        if ( *((_BYTE *)v34 + 16) <= 0x17u )
        {
          v74 = sub_146F1B0(v43, (__int64)v34);
          v47 = v87;
          v48 = (__int64 *)v74;
          v49 = (unsigned int)v90;
          if ( (unsigned int)v90 < HIDWORD(v90) )
            goto LABEL_23;
        }
        else
        {
          v44 = sub_145DC80(v43, (__int64)v34);
          v47 = v87;
          v48 = (__int64 *)v44;
          v49 = (unsigned int)v90;
          if ( (unsigned int)v90 < HIDWORD(v90) )
          {
LABEL_23:
            v89[v49] = v48;
            LODWORD(v90) = v90 + 1;
            if ( v33 != (__m128i *)v32 )
            {
              v50 = v47;
              do
              {
                if ( *(_QWORD *)v32 != v35 )
                  break;
                v52 = (unsigned int)v90;
                if ( (unsigned int)v90 >= HIDWORD(v90) )
                {
                  sub_16CD150((__int64)&v89, v91, 0, 8, v45, v46);
                  v52 = (unsigned int)v90;
                }
                v51 = *(__int64 **)(v32 + 8);
                v32 += 16LL;
                v89[v52] = v51;
                LODWORD(v90) = v90 + 1;
              }
              while ( v33 != (__m128i *)v32 );
              v47 = v50;
            }
            v61 = sub_3875200(a1, v47, *(double *)a3.m128_u64, *(double *)a4.m128i_i64, a5);
            v55 = v89;
            v56 = v86;
            v57 = (__int64 ***)v61;
            v58 = v85;
            v59 = &v89[(unsigned int)v90];
            goto LABEL_31;
          }
        }
        v83 = v47;
        sub_16CD150((__int64)&v89, v91, 0, 8, v45, v46);
        v49 = (unsigned int)v90;
        v47 = v83;
        goto LABEL_23;
      }
      v82 = v34;
      v37 = v33;
      v89 = v91;
      v90 = 0x400000000LL;
      v84 = v36;
      while ( *(_QWORD *)v32 == v35 )
      {
        v40 = *(_QWORD *)(v32 + 8);
        if ( *(_WORD *)(v40 + 24) == 10 && (v41 = *(_QWORD *)(v40 - 8), *(_BYTE *)(v41 + 16) <= 0x17u) )
        {
          v40 = sub_146F1B0(*a1, v41);
          v38 = (unsigned int)v90;
          if ( (unsigned int)v90 >= HIDWORD(v90) )
          {
LABEL_19:
            sub_16CD150((__int64)&v89, v91, 0, 8, v28, v29);
            v38 = (unsigned int)v90;
          }
        }
        else
        {
          v38 = (unsigned int)v90;
          if ( (unsigned int)v90 >= HIDWORD(v90) )
            goto LABEL_19;
        }
        v32 += 16LL;
        v89[v38] = (__int64 *)v40;
        v39 = (unsigned int)(v90 + 1);
        LODWORD(v90) = v90 + 1;
        if ( v37 == (__m128i *)v32 )
        {
          v33 = v37;
          v53 = v84;
          v54 = v82;
          goto LABEL_30;
        }
      }
      v33 = v37;
      v53 = v84;
      v39 = (unsigned int)v90;
      v54 = v82;
LABEL_30:
      v55 = v89;
      v56 = v86;
      v57 = v54;
      v58 = v53;
      v59 = &v89[v39];
LABEL_31:
      v34 = (__int64 ***)sub_3877100(a1, v55, v59, v58, v56, v57, (__m128i)a3, a4, a5, a6, v30, v31, a9, a10);
      if ( v89 != v91 )
        _libc_free((unsigned __int64)v89);
LABEL_33:
      if ( (__m128i *)v32 == v33 )
      {
        v32 = (unsigned __int64)v92;
        v27 = v34;
        break;
      }
    }
  }
  if ( (_BYTE *)v32 != v94 )
    _libc_free(v32);
  return v27;
}
