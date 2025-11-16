// Function: sub_202ACC0
// Address: 0x202acc0
//
__int64 *__fastcall sub_202ACC0(_QWORD *a1, unsigned __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // r15
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // r9
  __int64 v13; // r13
  __int64 v14; // rax
  char v15; // dl
  int v16; // eax
  __int64 *result; // rax
  __int64 v18; // rax
  _QWORD *v19; // r14
  __int64 v20; // r9
  __int64 v21; // rax
  char v22; // dl
  __int64 v23; // rax
  unsigned int v24; // eax
  _QWORD *v25; // r15
  unsigned __int64 v26; // r11
  __int64 v27; // rsi
  unsigned __int8 *v28; // r13
  const void **v29; // rbx
  __int64 v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rsi
  __int64 v34; // rax
  __int64 v35; // rdx
  char v36; // di
  __int64 v37; // rdx
  unsigned int v38; // eax
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  _QWORD *v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // r15
  __int64 v46; // r14
  __int64 v47; // rax
  unsigned int v48; // edx
  _QWORD *v49; // r13
  __int64 v50; // rax
  unsigned int v51; // eax
  unsigned int v52; // eax
  __int64 *v53; // rdi
  const void **v54; // rdx
  unsigned int v55; // edx
  __int128 v56; // [rsp-50h] [rbp-170h]
  __int64 v57; // [rsp+20h] [rbp-100h]
  __int64 v58; // [rsp+28h] [rbp-F8h]
  __int64 v59; // [rsp+30h] [rbp-F0h]
  __int64 v60; // [rsp+40h] [rbp-E0h]
  unsigned __int64 v61; // [rsp+40h] [rbp-E0h]
  __int64 *v62; // [rsp+40h] [rbp-E0h]
  _QWORD *v63; // [rsp+40h] [rbp-E0h]
  __int64 v64; // [rsp+40h] [rbp-E0h]
  __int64 v65; // [rsp+48h] [rbp-D8h]
  unsigned __int64 v66; // [rsp+48h] [rbp-D8h]
  __int64 *v67; // [rsp+50h] [rbp-D0h]
  __int64 v68; // [rsp+70h] [rbp-B0h] BYREF
  const void **v69; // [rsp+78h] [rbp-A8h]
  __int64 v70; // [rsp+80h] [rbp-A0h] BYREF
  int v71; // [rsp+88h] [rbp-98h]
  __int128 v72; // [rsp+90h] [rbp-90h] BYREF
  __int64 v73; // [rsp+A0h] [rbp-80h]
  __int64 v74; // [rsp+B0h] [rbp-70h] BYREF
  __int64 v75; // [rsp+B8h] [rbp-68h]
  __int64 v76; // [rsp+C0h] [rbp-60h]
  __int128 v77; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v78; // [rsp+E0h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 32);
  v8 = *(unsigned int *)(v7 + 8);
  v9 = *(_QWORD *)v7;
  v10 = _mm_loadu_si128((const __m128i *)v7);
  v11 = _mm_loadu_si128((const __m128i *)(v7 + 40));
  v12 = *(_QWORD *)(v7 + 40);
  v13 = *(unsigned int *)(v7 + 48);
  v14 = *(_QWORD *)(*(_QWORD *)v7 + 40LL) + 16 * v8;
  v15 = *(_BYTE *)v14;
  v69 = *(const void ***)(v14 + 8);
  v16 = *(unsigned __int16 *)(v12 + 24);
  LOBYTE(v68) = v15;
  if ( v16 != 10 && v16 != 32 )
  {
    if ( (unsigned __int8)sub_2016240(a1, a2, **(_BYTE **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), 1u, 0, 0) )
      return 0;
    v33 = *(_QWORD *)(a2 + 72);
    v70 = v33;
    if ( v33 )
      sub_1623A60((__int64)&v70, v33, 2);
    v71 = *(_DWORD *)(a2 + 64);
    LOBYTE(v34) = sub_1F7E0F0((__int64)&v68);
    v57 = v34;
    v58 = v35;
    if ( (_BYTE)v68 )
    {
      if ( (unsigned __int8)(v68 - 14) > 0x5Fu )
        goto LABEL_20;
    }
    else if ( !sub_1F58D20((__int64)&v68) )
    {
LABEL_20:
      v36 = v68;
      v37 = (__int64)v69;
LABEL_21:
      LOBYTE(v77) = v36;
      *((_QWORD *)&v77 + 1) = v37;
      if ( v36 )
        v38 = sub_2021900(v36);
      else
        v38 = sub_1F58D40((__int64)&v77);
      if ( v38 <= 7 )
      {
        v50 = v57;
        v58 = 0;
        LOBYTE(v50) = 3;
        v57 = v50;
        v51 = sub_1D15970(&v68);
        v52 = sub_1F7DEB0(*(_QWORD **)(a1[1] + 48LL), v57, 0, v51, 0);
        v53 = (__int64 *)a1[1];
        v69 = v54;
        LODWORD(v68) = v52;
        v9 = sub_1D309E0(
               v53,
               144,
               (__int64)&v70,
               v52,
               v54,
               0,
               *(double *)v10.m128i_i64,
               *(double *)v11.m128i_i64,
               *(double *)a5.m128i_i64,
               *(_OWORD *)&v10);
        v8 = v55;
      }
      v63 = sub_1D29C20((_QWORD *)a1[1], (unsigned int)v68, (__int64)v69, 1, v39, v40);
      v65 = v41;
      sub_1E341E0((__int64)&v72, *(_QWORD *)(a1[1] + 32LL), *((_DWORD *)v63 + 21), 0);
      v42 = (_QWORD *)a1[1];
      v77 = 0u;
      v78 = 0;
      v43 = sub_1D2BF40(
              v42,
              (__int64)(v42 + 11),
              0,
              (__int64)&v70,
              v9,
              v8 | v10.m128i_i64[1] & 0xFFFFFFFF00000000LL,
              (__int64)v63,
              v65,
              v72,
              v73,
              0,
              0,
              (__int64)&v77);
      v45 = v44;
      v46 = v43;
      v47 = sub_20BD400(*a1, a1[1], (_DWORD)v63, v65, v68, (_DWORD)v69, v11.m128i_i64[0], v11.m128i_i64[1]);
      v74 = 0;
      v64 = v47;
      v75 = 0;
      v76 = 0;
      v66 = v48 | v65 & 0xFFFFFFFF00000000LL;
      v49 = (_QWORD *)a1[1];
      sub_1E34280((__int64)&v77, v49[4]);
      *((_QWORD *)&v56 + 1) = v45;
      *(_QWORD *)&v56 = v46;
      result = (__int64 *)sub_1D2B810(
                            v49,
                            1u,
                            (__int64)&v70,
                            **(unsigned __int8 **)(a2 + 40),
                            *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                            0,
                            v56,
                            v64,
                            v66,
                            v77,
                            v78,
                            v57,
                            v58,
                            0,
                            (__int64)&v74);
      if ( v70 )
      {
        v67 = result;
        sub_161E7C0((__int64)&v70, v70);
        return v67;
      }
      return result;
    }
    v36 = sub_1F7E0F0((__int64)&v68);
    goto LABEL_21;
  }
  v18 = *(_QWORD *)(v12 + 88);
  v19 = *(_QWORD **)(v18 + 24);
  if ( *(_DWORD *)(v18 + 32) > 0x40u )
    v19 = (_QWORD *)*v19;
  v60 = v12;
  *(_QWORD *)&v72 = 0;
  DWORD2(v72) = 0;
  v74 = 0;
  LODWORD(v75) = 0;
  sub_2017DE0((__int64)a1, v10.m128i_u64[0], v10.m128i_i64[1], &v72, &v74);
  v20 = v60;
  v21 = *(_QWORD *)(v72 + 40) + 16LL * DWORD2(v72);
  v22 = *(_BYTE *)v21;
  v23 = *(_QWORD *)(v21 + 8);
  LOBYTE(v77) = v22;
  *((_QWORD *)&v77 + 1) = v23;
  if ( v22 )
  {
    v25 = (_QWORD *)a1[1];
    v26 = word_4305480[(unsigned __int8)(v22 - 14)];
    if ( v26 <= (unsigned __int64)v19 )
      goto LABEL_10;
    return sub_1D2DF70(v25, (__int64 *)a2, v72, *((__int64 *)&v72 + 1), v11.m128i_i64[0], v11.m128i_i64[1]);
  }
  v24 = sub_1F58D30((__int64)&v77);
  v20 = v60;
  v25 = (_QWORD *)a1[1];
  v26 = v24;
  if ( v24 > (unsigned __int64)v19 )
    return sub_1D2DF70(v25, (__int64 *)a2, v72, *((__int64 *)&v72 + 1), v11.m128i_i64[0], v11.m128i_i64[1]);
LABEL_10:
  v27 = *(_QWORD *)(a2 + 72);
  v28 = (unsigned __int8 *)(*(_QWORD *)(v20 + 40) + 16 * v13);
  v29 = (const void **)*((_QWORD *)v28 + 1);
  v30 = *v28;
  *(_QWORD *)&v77 = v27;
  if ( v27 )
  {
    v59 = v30;
    v61 = v26;
    sub_1623A60((__int64)&v77, v27, 2);
    v30 = v59;
    v26 = v61;
  }
  DWORD2(v77) = *(_DWORD *)(a2 + 64);
  v31 = sub_1D38BB0((__int64)v25, (__int64)v19 - v26, (__int64)&v77, v30, v29, 0, v10, *(double *)v11.m128i_i64, a5, 0);
  result = sub_1D2DF70(v25, (__int64 *)a2, v74, v75, v31, v32);
  if ( (_QWORD)v77 )
  {
    v62 = result;
    sub_161E7C0((__int64)&v77, v77);
    return v62;
  }
  return result;
}
