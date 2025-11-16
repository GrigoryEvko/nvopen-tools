// Function: sub_2021EF0
// Address: 0x2021ef0
//
__int64 __fastcall sub_2021EF0(
        __int64 *a1,
        __int64 a2,
        const void **a3,
        _QWORD *a4,
        unsigned int a5,
        int a6,
        __m128i a7,
        double a8,
        __m128i a9)
{
  unsigned int v9; // r13d
  unsigned int v10; // r14d
  __int64 v11; // rbx
  __int64 v12; // rax
  unsigned int *v13; // rax
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int8 v17; // r8
  __int64 v18; // rax
  char v19; // r8
  unsigned int v20; // ecx
  unsigned int v21; // eax
  const void **v22; // rdx
  __int64 v23; // rdx
  const void **v24; // r14
  unsigned int v25; // ebx
  __int64 v26; // rax
  unsigned int v27; // edx
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned int v31; // edx
  __int64 v32; // rax
  char v33; // bl
  __int64 v34; // rax
  unsigned int v35; // ecx
  unsigned __int32 v36; // r14d
  __int64 v37; // rax
  const void **v38; // rdx
  __int64 v39; // rcx
  unsigned int v40; // edx
  int v41; // eax
  unsigned int v42; // ecx
  __int64 v43; // r12
  unsigned int v45; // eax
  __int128 v46; // [rsp-30h] [rbp-130h]
  __int64 v47; // [rsp+10h] [rbp-F0h]
  _QWORD *v48; // [rsp+18h] [rbp-E8h]
  unsigned int v49; // [rsp+24h] [rbp-DCh]
  __int64 v50; // [rsp+28h] [rbp-D8h]
  __int64 v51; // [rsp+30h] [rbp-D0h]
  __int64 v52; // [rsp+38h] [rbp-C8h]
  unsigned int v53; // [rsp+38h] [rbp-C8h]
  unsigned int v54; // [rsp+38h] [rbp-C8h]
  _QWORD *v55; // [rsp+40h] [rbp-C0h]
  const void **v56; // [rsp+40h] [rbp-C0h]
  __int64 v57; // [rsp+40h] [rbp-C0h]
  unsigned int v58; // [rsp+48h] [rbp-B8h]
  int v59; // [rsp+48h] [rbp-B8h]
  __int64 (__fastcall *v60)(__int64, __int64); // [rsp+48h] [rbp-B8h]
  unsigned int v61; // [rsp+48h] [rbp-B8h]
  unsigned int v65; // [rsp+5Ch] [rbp-A4h]
  __int64 v66; // [rsp+60h] [rbp-A0h]
  unsigned __int32 v67; // [rsp+60h] [rbp-A0h]
  __int128 v68; // [rsp+60h] [rbp-A0h]
  char v69; // [rsp+60h] [rbp-A0h]
  __int64 v70; // [rsp+90h] [rbp-70h] BYREF
  const void **v71; // [rsp+98h] [rbp-68h]
  __int64 v72; // [rsp+A0h] [rbp-60h] BYREF
  int v73; // [rsp+A8h] [rbp-58h]
  __m128i v74; // [rsp+B0h] [rbp-50h] BYREF
  __m128i v75; // [rsp+C0h] [rbp-40h] BYREF

  v11 = 16LL * a5;
  v12 = a1[2];
  v70 = a2;
  v71 = a3;
  v50 = v12;
  v13 = (unsigned int *)(v11 + *a4);
  v14 = *(_QWORD *)v13;
  v15 = *(_QWORD *)(*(_QWORD *)v13 + 72LL);
  v72 = v15;
  if ( v15 )
  {
    v66 = v14;
    sub_1623A60((__int64)&v72, v15, 2);
    v14 = v66;
    v13 = (unsigned int *)(v11 + *a4);
  }
  v73 = *(_DWORD *)(v14 + 64);
  v16 = *(_QWORD *)(*(_QWORD *)v13 + 40LL) + 16LL * v13[2];
  v17 = *(_BYTE *)v16;
  v18 = *(_QWORD *)(v16 + 8);
  v74.m128i_i8[0] = v17;
  v74.m128i_i64[1] = v18;
  if ( (_BYTE)v70 )
  {
    v49 = sub_2021900(v70);
  }
  else
  {
    v69 = v17;
    v45 = sub_1F58D40((__int64)&v70);
    v19 = v69;
    v49 = v45;
  }
  if ( v19 )
    v20 = sub_2021900(v19);
  else
    v20 = sub_1F58D40((__int64)&v74);
  v67 = v74.m128i_i32[0];
  v58 = v49 / v20;
  v55 = (_QWORD *)a1[6];
  v52 = v74.m128i_i64[1];
  LOBYTE(v21) = sub_1D15020(v74.m128i_i8[0], v49 / v20);
  v22 = 0;
  if ( !(_BYTE)v21 )
  {
    v21 = sub_1F593D0(v55, v67, v52, v58);
    v10 = v21;
  }
  LOBYTE(v10) = v21;
  v56 = v22;
  v53 = v10;
  *(_QWORD *)&v68 = sub_1D309E0(
                      a1,
                      111,
                      (__int64)&v72,
                      v10,
                      v22,
                      0,
                      *(double *)a7.m128i_i64,
                      a8,
                      *(double *)a9.m128i_i64,
                      *(_OWORD *)(*a4 + v11));
  *((_QWORD *)&v68 + 1) = v23;
  v65 = a5 + 1;
  if ( v65 != a6 )
  {
    v24 = v56;
    v59 = 1;
    while ( 1 )
    {
      v57 = 16LL * v65;
      v32 = *(_QWORD *)(*(_QWORD *)(*a4 + v57) + 40LL) + 16LL * *(unsigned int *)(*a4 + v57 + 8);
      v33 = *(_BYTE *)v32;
      v34 = *(_QWORD *)(v32 + 8);
      v75.m128i_i8[0] = v33;
      v75.m128i_i64[1] = v34;
      if ( v74.m128i_i8[0] != v33 )
        break;
      if ( v74.m128i_i64[1] != v34 && !v33 )
        goto LABEL_34;
      v25 = v59;
LABEL_14:
      v60 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v50 + 48LL);
      v26 = sub_1E0A0C0(a1[4]);
      if ( v60 == sub_1D13A20 )
      {
        v27 = 8 * sub_15A9520(v26, 0);
        if ( v27 == 32 )
        {
          v28 = 5;
        }
        else if ( v27 > 0x20 )
        {
          v28 = 6;
          if ( v27 != 64 )
          {
            v28 = 0;
            if ( v27 == 128 )
              v28 = 7;
          }
        }
        else
        {
          v28 = 3;
          if ( v27 != 8 )
            v28 = 4 * (v27 == 16);
        }
      }
      else
      {
        v28 = v60(v50, v26);
      }
      LOBYTE(v9) = v28;
      v59 = v25 + 1;
      v29 = sub_1D38BB0((__int64)a1, v25, (__int64)&v72, v9, 0, 0, a7, a8, a9, 0);
      v46 = *(_OWORD *)(*a4 + 16LL * v65++);
      *(_QWORD *)&v68 = sub_1D3A900(
                          a1,
                          0x69u,
                          (__int64)&v72,
                          v53,
                          v24,
                          0,
                          (__m128)a7,
                          a8,
                          a9,
                          v68,
                          *((__int16 **)&v68 + 1),
                          v46,
                          v29,
                          v30);
      *((_QWORD *)&v68 + 1) = v31 | *((_QWORD *)&v68 + 1) & 0xFFFFFFFF00000000LL;
      if ( a6 == v65 )
        goto LABEL_38;
    }
    if ( v33 )
      v35 = sub_2021900(v33);
    else
LABEL_34:
      v35 = sub_1F58D40((__int64)&v75);
    v36 = v75.m128i_i32[0];
    v54 = v49 / v35;
    v48 = (_QWORD *)a1[6];
    v47 = v75.m128i_i64[1];
    LOBYTE(v37) = sub_1D15020(v75.m128i_i8[0], v49 / v35);
    v38 = 0;
    if ( !(_BYTE)v37 )
    {
      v37 = sub_1F593D0(v48, v36, v47, v54);
      v51 = v37;
    }
    v39 = v51;
    v24 = v38;
    LOBYTE(v39) = v37;
    v51 = v39;
    v53 = v39;
    *(_QWORD *)&v68 = sub_1D309E0(
                        a1,
                        158,
                        (__int64)&v72,
                        (unsigned int)v39,
                        v38,
                        0,
                        *(double *)a7.m128i_i64,
                        a8,
                        *(double *)a9.m128i_i64,
                        v68);
    *((_QWORD *)&v68 + 1) = v40 | *((_QWORD *)&v68 + 1) & 0xFFFFFFFF00000000LL;
    if ( v74.m128i_i8[0] )
      v41 = sub_2021900(v74.m128i_i8[0]);
    else
      v41 = sub_1F58D40((__int64)&v74);
    v61 = v59 * v41;
    if ( v33 )
      v42 = sub_2021900(v33);
    else
      v42 = sub_1F58D40((__int64)&v75);
    a7 = _mm_loadu_si128(&v75);
    v74 = a7;
    v25 = v61 / v42;
    goto LABEL_14;
  }
LABEL_38:
  v43 = sub_1D309E0(
          a1,
          158,
          (__int64)&v72,
          (unsigned int)v70,
          v71,
          0,
          *(double *)a7.m128i_i64,
          a8,
          *(double *)a9.m128i_i64,
          v68);
  if ( v72 )
    sub_161E7C0((__int64)&v72, v72);
  return v43;
}
