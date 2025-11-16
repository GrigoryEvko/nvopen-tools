// Function: sub_3270620
// Address: 0x3270620
//
__int64 __fastcall sub_3270620(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        unsigned int a11)
{
  __m128i v14; // xmm0
  __int64 v15; // rsi
  __int64 v16; // r15
  _QWORD *v17; // rdi
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r12
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v27; // rdi
  __int64 v28; // r15
  __int64 v29; // rcx
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r9
  __int128 v36; // xmm1
  __int64 v37; // r15
  __int64 v38; // rax
  __int128 v39; // rax
  int v40; // r9d
  __int128 v41; // [rsp-20h] [rbp-1F0h]
  __int64 v42; // [rsp-10h] [rbp-1E0h]
  __int64 v43; // [rsp-10h] [rbp-1E0h]
  __int64 v44; // [rsp-8h] [rbp-1D8h]
  __int64 v45; // [rsp+8h] [rbp-1C8h]
  __int64 v46; // [rsp+10h] [rbp-1C0h]
  __int64 v50; // [rsp+30h] [rbp-1A0h]
  int v51; // [rsp+30h] [rbp-1A0h]
  unsigned int v52; // [rsp+40h] [rbp-190h]
  int v53; // [rsp+44h] [rbp-18Ch]
  __int64 v54; // [rsp+48h] [rbp-188h]
  __m128i v55; // [rsp+50h] [rbp-180h] BYREF
  __int64 v56; // [rsp+60h] [rbp-170h]
  __int64 v57; // [rsp+68h] [rbp-168h]
  __int64 v58; // [rsp+70h] [rbp-160h]
  __int64 v59; // [rsp+78h] [rbp-158h]
  _QWORD v60[8]; // [rsp+80h] [rbp-150h] BYREF
  __int64 v61; // [rsp+C0h] [rbp-110h]
  int v62; // [rsp+C8h] [rbp-108h]
  __int64 v63; // [rsp+D0h] [rbp-100h]
  __int64 v64; // [rsp+D8h] [rbp-F8h]
  __int64 v65; // [rsp+E0h] [rbp-F0h] BYREF
  int v66; // [rsp+E8h] [rbp-E8h]
  _QWORD *v67; // [rsp+F0h] [rbp-E0h]
  __int64 v68; // [rsp+F8h] [rbp-D8h]
  __int64 v69; // [rsp+100h] [rbp-D0h] BYREF
  _QWORD v70[8]; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v71; // [rsp+150h] [rbp-80h]
  int v72; // [rsp+158h] [rbp-78h]
  __int64 v73; // [rsp+160h] [rbp-70h]
  __int64 v74; // [rsp+168h] [rbp-68h]
  __int64 v75; // [rsp+170h] [rbp-60h] BYREF
  int v76; // [rsp+178h] [rbp-58h]
  _QWORD *v77; // [rsp+180h] [rbp-50h]
  __int64 v78; // [rsp+188h] [rbp-48h]
  __int64 v79; // [rsp+190h] [rbp-40h] BYREF

  v14 = _mm_loadu_si128((const __m128i *)&a7);
  v15 = a8;
  v54 = a9;
  v55 = v14;
  v16 = *(_QWORD *)a1;
  v52 = a11;
  v53 = a10;
  v17 = *(_QWORD **)(a1 + 8);
  if ( ((_DWORD)a10 != v14.m128i_i32[2] || a5 != (_QWORD)a8 || (_DWORD)a6 != DWORD2(a8) || a9 != v14.m128i_i64[0])
    && ((_DWORD)a6 != (_DWORD)a10 || (_QWORD)a8 != v14.m128i_i64[0] || DWORD2(a8) != v14.m128i_i32[2] || a9 != a5) )
  {
    LODWORD(v70[0]) = 2;
    v18 = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, _QWORD, __int64, _QWORD, _QWORD, _QWORD *, _QWORD))(*v17 + 2264LL))(
            v17,
            a8,
            *((_QWORD *)&a8 + 1),
            v16,
            *(unsigned __int8 *)(a1 + 33),
            *(unsigned __int8 *)(a1 + 35),
            v70,
            0);
    v21 = v18;
    if ( v18 )
    {
      if ( SLODWORD(v70[0]) <= 1 )
      {
        v22 = (unsigned int)v19;
        v50 = v19;
        v23 = sub_33ECD10(1, v15, v19, v20, v42, v44);
        v69 = 0;
        v60[6] = v23;
        v61 = 0x100000000LL;
        v60[7] = 0;
        v62 = 0;
        v63 = 0;
        v64 = 0xFFFFFFFFLL;
        v68 = 0;
        v67 = v60;
        v58 = v21;
        v59 = v22;
        v65 = v21;
        v66 = v22;
        v24 = *(_QWORD *)(v21 + 56);
        memset(v60, 0, 24);
        v60[3] = 328;
        v60[4] = -65536;
        v69 = v24;
        if ( v24 )
          *(_QWORD *)(v24 + 24) = &v69;
        v68 = v21 + 56;
        v60[5] = &v65;
        *(_QWORD *)(v21 + 56) = &v65;
        LODWORD(v61) = 1;
        if ( a5 == v21 && (_DWORD)a6 == (_DWORD)v50 )
        {
          v27 = *(_QWORD *)(a1 + 8);
          v28 = *(_QWORD *)a1;
          v46 = v50;
          v29 = *(_QWORD *)a1;
          v30 = *(unsigned __int8 *)(a1 + 35);
          LODWORD(v70[0]) = 2;
          v31 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, __int64, _QWORD *))(*(_QWORD *)v27 + 2264LL))(
                  v27,
                  v55.m128i_i64[0],
                  v55.m128i_i64[1],
                  v29,
                  *(unsigned __int8 *)(a1 + 33),
                  v30,
                  v70);
          if ( v31 )
          {
            if ( SLODWORD(v70[0]) <= 1 )
            {
              v37 = (unsigned int)v32;
              v51 = v32;
              v45 = v31;
              v38 = sub_33ECD10(1, v31, v32, v43, v33, v34);
              v79 = 0;
              v71 = 0x100000000LL;
              v70[6] = v38;
              v74 = 0xFFFFFFFFLL;
              memset(v70, 0, 24);
              v70[3] = 328;
              v70[4] = -65536;
              v70[7] = 0;
              v72 = 0;
              v73 = 0;
              v78 = 0;
              v77 = v70;
              v56 = v45;
              v57 = v37;
              v75 = v45;
              v76 = v37;
              v79 = *(_QWORD *)(v45 + 56);
              if ( v79 )
                *(_QWORD *)(v79 + 24) = &v79;
              v78 = v45 + 56;
              *(_QWORD *)(v45 + 56) = &v75;
              LODWORD(v71) = 1;
              v70[5] = &v75;
              if ( v54 == v45 && v53 == v51 )
              {
                *((_QWORD *)&v41 + 1) = v52;
                *(_QWORD *)&v41 = v46;
                *(_QWORD *)&v39 = sub_326B770(
                                    a2,
                                    a3,
                                    a4,
                                    a5,
                                    a6,
                                    v21,
                                    *(_OWORD *)&v55,
                                    v41,
                                    *(_QWORD **)(a1 + 8),
                                    *(_QWORD *)a1);
                if ( (_QWORD)v39 )
                {
                  v25 = sub_33FAF80(*(_QWORD *)a1, 244, a2, a3, a4, v40, v39);
                  sub_33CF710(v70);
                  goto LABEL_11;
                }
              }
              sub_33CF710(v70);
            }
            else if ( !*(_QWORD *)(v31 + 56) )
            {
              sub_33ECEA0(v28, v31);
            }
          }
        }
        v25 = 0;
LABEL_11:
        sub_33CF710(v60);
        return v25;
      }
      if ( !*(_QWORD *)(v18 + 56) )
        sub_33ECEA0(v16, v18);
    }
    return 0;
  }
  a10 = v16;
  v35 = a8;
  v36 = (__int128)_mm_load_si128(&v55);
  LODWORD(a8) = DWORD2(a8);
  DWORD2(a8) = a11;
  return sub_326B770(a2, a3, a4, a5, a6, v35, v36, a8, v17, v16);
}
