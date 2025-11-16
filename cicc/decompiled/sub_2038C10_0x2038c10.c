// Function: sub_2038C10
// Address: 0x2038c10
//
__int64 __fastcall sub_2038C10(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  unsigned int v5; // r14d
  const __m128i *v8; // rax
  __int64 v9; // rsi
  __m128i v10; // xmm0
  __int64 v11; // rcx
  __int64 v12; // rsi
  __int64 v13; // rdx
  unsigned int v14; // eax
  const void **v15; // rdx
  __int64 v16; // rax
  char v17; // dl
  __int64 v18; // rax
  unsigned int v19; // eax
  const void **v20; // rdx
  unsigned __int8 v21; // dl
  __int64 v22; // r9
  __int64 *v23; // r15
  unsigned __int64 v24; // rax
  __int64 v25; // r12
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // r8d
  unsigned __int64 v29; // r11
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned int v32; // edx
  char v33; // al
  __int128 v34; // rax
  __int128 v35; // rax
  __int64 *v36; // rdi
  __int64 v37; // r11
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 *i; // rdi
  int v41; // r8d
  _QWORD *v42; // r14
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // rdx
  _QWORD *v46; // rdx
  __int64 v47; // r14
  __int64 v49; // r10
  unsigned int v50; // eax
  unsigned int v51; // edx
  unsigned __int64 v52; // rax
  __int64 v53; // rdx
  char v54; // di
  __int64 v55; // rax
  int v56; // eax
  char v57; // di
  int v58; // r12d
  int v59; // eax
  int v60; // eax
  __int128 v61; // [rsp-10h] [rbp-240h]
  __int64 v62; // [rsp+18h] [rbp-218h]
  unsigned __int64 v63; // [rsp+20h] [rbp-210h]
  unsigned int v64; // [rsp+28h] [rbp-208h]
  const void **v65; // [rsp+30h] [rbp-200h]
  unsigned int v66; // [rsp+38h] [rbp-1F8h]
  __int16 v67; // [rsp+3Eh] [rbp-1F2h]
  __int64 v68; // [rsp+40h] [rbp-1F0h]
  unsigned int v69; // [rsp+48h] [rbp-1E8h]
  __int64 v70; // [rsp+48h] [rbp-1E8h]
  unsigned int v71; // [rsp+50h] [rbp-1E0h]
  __int64 (__fastcall *v72)(__int64, __int64); // [rsp+50h] [rbp-1E0h]
  __int64 v73; // [rsp+50h] [rbp-1E0h]
  __int64 v74; // [rsp+58h] [rbp-1D8h]
  __int64 v75; // [rsp+58h] [rbp-1D8h]
  unsigned __int64 v76; // [rsp+58h] [rbp-1D8h]
  unsigned int v77; // [rsp+60h] [rbp-1D0h]
  const void **v78; // [rsp+68h] [rbp-1C8h]
  unsigned __int64 v79; // [rsp+78h] [rbp-1B8h]
  __int64 v80; // [rsp+B0h] [rbp-180h] BYREF
  int v81; // [rsp+B8h] [rbp-178h]
  __int64 v82; // [rsp+C0h] [rbp-170h] BYREF
  const void **v83; // [rsp+C8h] [rbp-168h]
  char v84[8]; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v85; // [rsp+D8h] [rbp-158h]
  __int64 v86; // [rsp+E0h] [rbp-150h] BYREF
  int v87; // [rsp+E8h] [rbp-148h]
  _QWORD *v88; // [rsp+F0h] [rbp-140h] BYREF
  __int64 v89; // [rsp+F8h] [rbp-138h]
  _QWORD v90[38]; // [rsp+100h] [rbp-130h] BYREF

  v67 = *(_WORD *)(a2 + 24);
  v8 = *(const __m128i **)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 72);
  v10 = _mm_loadu_si128(v8);
  v11 = v8->m128i_i64[0];
  LODWORD(v8) = v8->m128i_i32[2];
  v80 = v9;
  v68 = v11;
  v71 = (unsigned int)v8;
  v79 = v10.m128i_u64[1];
  if ( v9 )
    sub_1623A60((__int64)&v80, v9, 2);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL);
  v81 = *(_DWORD *)(a2 + 64);
  sub_1F40D10((__int64)&v88, v12, v13, **(unsigned __int8 **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  LOBYTE(v82) = v89;
  v83 = (const void **)v90[0];
  LOBYTE(v14) = sub_1F7E0F0((__int64)&v82);
  v77 = v14;
  v78 = v15;
  if ( (_BYTE)v82 )
    v66 = word_4305480[(unsigned __int8)(v82 - 14)];
  else
    v66 = sub_1F58D30((__int64)&v82);
  v16 = *(_QWORD *)(v68 + 40) + 16LL * v71;
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v84[0] = v17;
  v85 = v18;
  LOBYTE(v19) = sub_1F7E0F0((__int64)v84);
  v65 = v20;
  v21 = v84[0];
  v64 = v19;
  if ( v84[0] )
  {
    v69 = word_4305480[(unsigned __int8)(v84[0] - 14)];
  }
  else
  {
    v50 = sub_1F58D30((__int64)v84);
    v21 = 0;
    v69 = v50;
  }
  sub_1F40D10((__int64)&v88, *(_QWORD *)a1, *(_QWORD *)(*(_QWORD *)(a1 + 8) + 48LL), v21, v85);
  if ( (_BYTE)v88 != 7 )
  {
    v23 = *(__int64 **)(a1 + 8);
    goto LABEL_9;
  }
  v68 = sub_20363F0(a1, v10.m128i_u64[0], v10.m128i_i64[1]);
  v71 = v51;
  v52 = v51 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v53 = *(_QWORD *)(v68 + 40) + 16LL * v51;
  v54 = *(_BYTE *)v53;
  v79 = v52;
  v55 = *(_QWORD *)(v53 + 8);
  v84[0] = v54;
  v85 = v55;
  if ( v54 )
  {
    v60 = sub_2021900(v54);
    v57 = v82;
    v58 = v60;
    if ( (_BYTE)v82 )
      goto LABEL_53;
  }
  else
  {
    v56 = sub_1F58D40((__int64)v84);
    v57 = v82;
    v58 = v56;
    if ( (_BYTE)v82 )
    {
LABEL_53:
      v59 = sub_2021900(v57);
      goto LABEL_54;
    }
  }
  v59 = sub_1F58D40((__int64)&v82);
LABEL_54:
  v23 = *(__int64 **)(a1 + 8);
  if ( v59 == v58 )
  {
    switch ( v67 )
    {
      case 150:
        v47 = sub_1D327E0(v23, v68, v79, (__int64)&v80, v82, v83, *(double *)v10.m128i_i64, a4, *(double *)a5.m128i_i64);
        goto LABEL_46;
      case 151:
        v47 = sub_1D32810(
                *(__int64 **)(a1 + 8),
                v68,
                v79,
                (__int64)&v80,
                v82,
                v83,
                *(double *)v10.m128i_i64,
                a4,
                *(double *)a5.m128i_i64);
        goto LABEL_46;
      case 149:
        v47 = sub_1D327B0(
                *(__int64 **)(a1 + 8),
                v68,
                v79,
                (__int64)&v80,
                v82,
                v83,
                *(double *)v10.m128i_i64,
                a4,
                *(double *)a5.m128i_i64);
        goto LABEL_46;
    }
  }
LABEL_9:
  v88 = v90;
  v89 = 0x1000000000LL;
  v24 = v69;
  if ( v69 > v66 )
    v24 = v66;
  if ( (_DWORD)v24 )
  {
    v63 = v24;
    v25 = 0;
    v62 = v71;
    while ( 1 )
    {
      v70 = *(_QWORD *)a1;
      v72 = *(__int64 (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 48LL);
      v31 = sub_1E0A0C0(v23[4]);
      if ( v72 == sub_1D13A20 )
      {
        v32 = 8 * sub_15A9520(v31, 0);
        if ( v32 == 32 )
        {
          v33 = 5;
        }
        else if ( v32 > 0x20 )
        {
          v33 = 6;
          if ( v32 != 64 )
          {
            v33 = 0;
            if ( v32 == 128 )
              v33 = 7;
          }
        }
        else
        {
          v33 = 3;
          if ( v32 != 8 )
            v33 = 4 * (v32 == 16);
        }
      }
      else
      {
        v33 = v72(v70, v31);
      }
      LOBYTE(v5) = v33;
      *(_QWORD *)&v34 = sub_1D38BB0((__int64)v23, v25, (__int64)&v80, v5, 0, 0, v10, a4, a5, 0);
      v79 = v62 | v79 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v35 = sub_1D332F0(
                          v23,
                          106,
                          (__int64)&v80,
                          v64,
                          v65,
                          0,
                          *(double *)v10.m128i_i64,
                          a4,
                          a5,
                          v68,
                          v79,
                          v34);
      v36 = *(__int64 **)(a1 + 8);
      if ( v67 == 150 )
      {
        v75 = *((_QWORD *)&v35 + 1);
        v38 = sub_1D309E0(
                v36,
                142,
                (__int64)&v80,
                v77,
                v78,
                0,
                *(double *)v10.m128i_i64,
                a4,
                *(double *)a5.m128i_i64,
                v35);
        v37 = v75;
        v49 = v38;
        v27 = (unsigned int)v27;
      }
      else
      {
        v74 = *((_QWORD *)&v35 + 1);
        if ( v67 == 151 )
          v26 = sub_1D309E0(
                  v36,
                  143,
                  (__int64)&v80,
                  v77,
                  v78,
                  0,
                  *(double *)v10.m128i_i64,
                  a4,
                  *(double *)a5.m128i_i64,
                  v35);
        else
          v26 = sub_1D309E0(
                  v36,
                  144,
                  (__int64)&v80,
                  v77,
                  v78,
                  0,
                  *(double *)v10.m128i_i64,
                  a4,
                  *(double *)a5.m128i_i64,
                  v35);
        v37 = v74;
        v49 = v26;
        v27 = (unsigned int)v27;
      }
      v29 = v27 | v37 & 0xFFFFFFFF00000000LL;
      v30 = (unsigned int)v89;
      if ( (unsigned int)v89 >= HIDWORD(v89) )
      {
        v73 = v49;
        v76 = v29;
        sub_16CD150((__int64)&v88, v90, 0, 16, v28, v22);
        v30 = (unsigned int)v89;
        v49 = v73;
        v29 = v76;
      }
      v24 = (unsigned __int64)&v88[2 * v30];
      ++v25;
      *(_QWORD *)v24 = v49;
      *(_QWORD *)(v24 + 8) = v29;
      LODWORD(v24) = v89 + 1;
      LODWORD(v89) = v89 + 1;
      if ( v63 == v25 )
        break;
      v23 = *(__int64 **)(a1 + 8);
    }
    v23 = *(__int64 **)(a1 + 8);
    v39 = (unsigned int)v24;
  }
  else
  {
    v39 = 0;
  }
  if ( v66 != (_DWORD)v24 )
  {
    for ( i = v23; ; i = *(__int64 **)(a1 + 8) )
    {
      v86 = 0;
      v87 = 0;
      v42 = sub_1D2B300(i, 0x30u, (__int64)&v86, v77, (__int64)v78, v22);
      v44 = v43;
      if ( v86 )
        sub_161E7C0((__int64)&v86, v86);
      v45 = (unsigned int)v89;
      if ( (unsigned int)v89 >= HIDWORD(v89) )
      {
        sub_16CD150((__int64)&v88, v90, 0, 16, v41, v22);
        v45 = (unsigned int)v89;
      }
      v46 = &v88[2 * v45];
      *v46 = v42;
      v46[1] = v44;
      LODWORD(v89) = v89 + 1;
      if ( v66 == (_DWORD)v89 )
        break;
    }
    v39 = v66;
    v23 = *(__int64 **)(a1 + 8);
  }
  *((_QWORD *)&v61 + 1) = v39;
  *(_QWORD *)&v61 = v88;
  v47 = (__int64)sub_1D359D0(v23, 104, (__int64)&v80, v82, v83, 0, *(double *)v10.m128i_i64, a4, a5, v61);
  if ( v88 != v90 )
    _libc_free((unsigned __int64)v88);
LABEL_46:
  if ( v80 )
    sub_161E7C0((__int64)&v80, v80);
  return v47;
}
