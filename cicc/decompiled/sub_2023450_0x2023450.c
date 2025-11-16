// Function: sub_2023450
// Address: 0x2023450
//
unsigned __int64 __fastcall sub_2023450(__int64 **a1, __int64 a2, __m128i *a3, _DWORD *a4, __m128i a5)
{
  char *v8; // rax
  __int64 v9; // rsi
  char v10; // dl
  __int64 v11; // rax
  __m128i v12; // xmm1
  __m128i v13; // xmm2
  __int64 v14; // rsi
  __int64 *v15; // rdx
  __int64 *v16; // rsi
  unsigned int *v17; // rax
  __int64 v18; // r15
  char v19; // di
  unsigned int v20; // eax
  __int64 *v21; // rdx
  _QWORD *v22; // r8
  char v23; // r15
  unsigned int v24; // eax
  unsigned int v25; // r14d
  unsigned int v26; // eax
  _QWORD *v27; // r8
  __int64 *v28; // rdx
  char v29; // r11
  unsigned int v30; // eax
  _BYTE *v31; // rax
  char v32; // r11
  const void **v33; // rax
  __int64 v34; // rax
  unsigned int v35; // r8d
  unsigned __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rcx
  const void **v39; // r8
  __int32 v40; // edx
  __int64 v41; // rax
  __int64 v42; // rsi
  unsigned int v43; // edx
  unsigned __int64 result; // rax
  unsigned int v45; // eax
  const void **v46; // rdx
  unsigned int v47; // eax
  const void **v48; // rdx
  __int64 v49; // r15
  __int8 v50; // al
  __int64 v51; // rdx
  bool v52; // al
  __int64 v53; // rax
  __int64 v54; // rcx
  const void **v55; // r8
  __int32 v56; // edx
  unsigned int v57; // edx
  __int64 v58; // rax
  __int64 v59; // rcx
  const void **v60; // r8
  __int32 v61; // edx
  unsigned int v62; // edx
  unsigned int v63; // [rsp+0h] [rbp-150h]
  char v64; // [rsp+0h] [rbp-150h]
  unsigned int v65; // [rsp+0h] [rbp-150h]
  unsigned int v66; // [rsp+10h] [rbp-140h]
  unsigned int v67; // [rsp+10h] [rbp-140h]
  char v68; // [rsp+10h] [rbp-140h]
  _QWORD *v69; // [rsp+10h] [rbp-140h]
  const void **v70; // [rsp+28h] [rbp-128h]
  __int64 v71; // [rsp+30h] [rbp-120h]
  unsigned __int64 v72; // [rsp+38h] [rbp-118h]
  unsigned __int64 v73; // [rsp+40h] [rbp-110h]
  __int64 *v74; // [rsp+40h] [rbp-110h]
  const void **v75; // [rsp+40h] [rbp-110h]
  __m128i v77; // [rsp+D0h] [rbp-80h] BYREF
  __m128i v78; // [rsp+E0h] [rbp-70h] BYREF
  __int64 v79; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v80; // [rsp+F8h] [rbp-58h]
  __m128i v81; // [rsp+100h] [rbp-50h] BYREF
  __m128i v82[4]; // [rsp+110h] [rbp-40h] BYREF

  v8 = *(char **)(a2 + 40);
  v77.m128i_i64[1] = 0;
  v78.m128i_i64[1] = 0;
  v9 = (__int64)a1[1];
  v77.m128i_i8[0] = 0;
  v78.m128i_i8[0] = 0;
  v10 = *v8;
  v11 = *((_QWORD *)v8 + 1);
  LOBYTE(v79) = v10;
  v80 = v11;
  sub_1D19A30((__int64)&v81, v9, &v79);
  v12 = _mm_loadu_si128(&v81);
  v13 = _mm_loadu_si128(v82);
  v14 = *(_QWORD *)(a2 + 72);
  v77 = v12;
  v79 = v14;
  v78 = v13;
  if ( v14 )
    sub_1623A60((__int64)&v79, v14, 2);
  v15 = a1[1];
  v16 = *a1;
  LODWORD(v80) = *(_DWORD *)(a2 + 64);
  v17 = *(unsigned int **)(a2 + 32);
  v72 = *(_QWORD *)v17;
  v18 = 16LL * v17[2];
  v73 = *(_QWORD *)v17;
  v71 = *((_QWORD *)v17 + 1);
  sub_1F40D10(
    (__int64)&v81,
    (__int64)v16,
    v15[6],
    *(unsigned __int8 *)(v18 + *(_QWORD *)(*(_QWORD *)v17 + 40LL)),
    *(_QWORD *)(v18 + *(_QWORD *)(*(_QWORD *)v17 + 40LL) + 8));
  if ( v81.m128i_i8[0] != 4 )
  {
    if ( v81.m128i_i8[0] == 6 )
    {
      sub_2017DE0((__int64)a1, v72, v71, a3, a4);
      v58 = sub_1D309E0(
              a1[1],
              158,
              (__int64)&v79,
              v77.m128i_u32[0],
              (const void **)v77.m128i_i64[1],
              0,
              *(double *)a5.m128i_i64,
              *(double *)v12.m128i_i64,
              *(double *)v13.m128i_i64,
              (__int128)*a3);
      v59 = v78.m128i_u32[0];
      v60 = (const void **)v78.m128i_i64[1];
      a3->m128i_i64[0] = v58;
      a3->m128i_i32[2] = v61;
      *(_QWORD *)a4 = sub_1D309E0(
                        a1[1],
                        158,
                        (__int64)&v79,
                        v59,
                        v60,
                        0,
                        *(double *)a5.m128i_i64,
                        *(double *)v12.m128i_i64,
                        *(double *)v13.m128i_i64,
                        *(_OWORD *)a4);
      result = v62;
      a4[2] = v62;
      goto LABEL_49;
    }
    v19 = v77.m128i_i8[0];
    if ( v81.m128i_i8[0] != 2 )
      goto LABEL_6;
  }
  v19 = v77.m128i_i8[0];
  if ( v77.m128i_i8[0] != v78.m128i_i8[0] )
  {
LABEL_6:
    if ( v19 )
    {
      v20 = sub_2021900(v19);
      goto LABEL_8;
    }
LABEL_28:
    v20 = sub_1F58D40((__int64)&v77);
LABEL_8:
    v21 = a1[1];
    v22 = (_QWORD *)v21[6];
    if ( v20 == 32 )
    {
      v23 = 5;
    }
    else if ( v20 > 0x20 )
    {
      if ( v20 == 64 )
      {
        v23 = 6;
      }
      else
      {
        if ( v20 != 128 )
        {
LABEL_35:
          v47 = sub_1F58CC0((_QWORD *)v21[6], v20);
          v70 = v48;
          v21 = a1[1];
          v23 = v47;
          v66 = v47;
          v22 = (_QWORD *)v21[6];
LABEL_13:
          v24 = v66;
          LOBYTE(v24) = v23;
          v25 = v24;
          if ( v78.m128i_i8[0] )
          {
            v74 = v21;
            v26 = sub_2021900(v78.m128i_i8[0]);
          }
          else
          {
            v69 = v22;
            v74 = v21;
            v26 = sub_1F58D40((__int64)&v78);
            v27 = v69;
          }
          v28 = v74;
          if ( v26 == 32 )
          {
            v29 = 5;
          }
          else if ( v26 > 0x20 )
          {
            if ( v26 == 64 )
            {
              v29 = 6;
            }
            else
            {
              if ( v26 != 128 )
              {
LABEL_33:
                v45 = sub_1F58CC0(v27, v26);
                v75 = v46;
                v29 = v45;
                v28 = a1[1];
                v63 = v45;
LABEL_20:
                v30 = v63;
                v64 = v29;
                LOBYTE(v30) = v29;
                v67 = v30;
                v31 = (_BYTE *)sub_1E0A0C0(v28[4]);
                v32 = v64;
                if ( *v31 )
                {
                  v33 = v70;
                  v25 = v67;
                  v70 = v75;
                  v75 = v33;
                  LOBYTE(v33) = v23;
                  v23 = v64;
                  v32 = (char)v33;
                }
                v65 = v67;
                LOBYTE(v25) = v23;
                v68 = v32;
                v34 = sub_200D2A0(
                        (__int64)a1,
                        v72,
                        v71,
                        *(double *)a5.m128i_i64,
                        *(double *)v12.m128i_i64,
                        *(double *)v13.m128i_i64);
                v35 = v65;
                LOBYTE(v35) = v68;
                sub_200E3C0(a1, v34, v36, v25, v70, (__int64)a3, a5, *(double *)v12.m128i_i64, v13, v35, v75, a4);
                if ( *(_BYTE *)sub_1E0A0C0(a1[1][4]) )
                {
                  a5 = _mm_loadu_si128(a3);
                  a3->m128i_i64[0] = *(_QWORD *)a4;
                  a3->m128i_i32[2] = a4[2];
                  *(_QWORD *)a4 = a5.m128i_i64[0];
                  a4[2] = a5.m128i_i32[2];
                }
                v37 = sub_1D309E0(
                        a1[1],
                        158,
                        (__int64)&v79,
                        v77.m128i_u32[0],
                        (const void **)v77.m128i_i64[1],
                        0,
                        *(double *)a5.m128i_i64,
                        *(double *)v12.m128i_i64,
                        *(double *)v13.m128i_i64,
                        (__int128)*a3);
                v38 = v78.m128i_u32[0];
                v39 = (const void **)v78.m128i_i64[1];
                a3->m128i_i64[0] = v37;
                a3->m128i_i32[2] = v40;
                v41 = sub_1D309E0(
                        a1[1],
                        158,
                        (__int64)&v79,
                        v38,
                        v39,
                        0,
                        *(double *)a5.m128i_i64,
                        *(double *)v12.m128i_i64,
                        *(double *)v13.m128i_i64,
                        *(_OWORD *)a4);
                v42 = v79;
                *(_QWORD *)a4 = v41;
                result = v43;
                a4[2] = v43;
                if ( v42 )
                  return sub_161E7C0((__int64)&v79, v42);
                return result;
              }
              v29 = 7;
            }
          }
          else if ( v26 == 8 )
          {
            v29 = 3;
          }
          else
          {
            v29 = 4;
            if ( v26 != 16 )
            {
              v29 = 2;
              if ( v26 != 1 )
                goto LABEL_33;
            }
          }
          v75 = 0;
          goto LABEL_20;
        }
        v23 = 7;
      }
    }
    else if ( v20 == 8 )
    {
      v23 = 3;
    }
    else
    {
      v23 = 4;
      if ( v20 != 16 )
      {
        v23 = 2;
        if ( v20 != 1 )
          goto LABEL_35;
      }
    }
    v70 = 0;
    goto LABEL_13;
  }
  if ( !v77.m128i_i8[0] && v78.m128i_i64[1] != v77.m128i_i64[1] )
    goto LABEL_28;
  v49 = *(_QWORD *)(v73 + 40) + v18;
  v50 = *(_BYTE *)v49;
  v51 = *(_QWORD *)(v49 + 8);
  v81.m128i_i8[0] = v50;
  v81.m128i_i64[1] = v51;
  if ( v50 )
    v52 = (unsigned __int8)(v50 - 2) <= 5u || (unsigned __int8)(v50 - 14) <= 0x47u;
  else
    v52 = sub_1F58CF0((__int64)&v81);
  if ( v52 )
    sub_20174B0((__int64)a1, v72, v71, a3, a4);
  else
    sub_2016B80((__int64)a1, v72, v71, a3, a4);
  if ( *(_BYTE *)sub_1E0A0C0(a1[1][4]) )
  {
    a5 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = a4[2];
    *(_QWORD *)a4 = a5.m128i_i64[0];
    a4[2] = a5.m128i_i32[2];
  }
  v53 = sub_1D309E0(
          a1[1],
          158,
          (__int64)&v79,
          v77.m128i_u32[0],
          (const void **)v77.m128i_i64[1],
          0,
          *(double *)a5.m128i_i64,
          *(double *)v12.m128i_i64,
          *(double *)v13.m128i_i64,
          (__int128)*a3);
  v54 = v78.m128i_u32[0];
  v55 = (const void **)v78.m128i_i64[1];
  a3->m128i_i64[0] = v53;
  a3->m128i_i32[2] = v56;
  *(_QWORD *)a4 = sub_1D309E0(
                    a1[1],
                    158,
                    (__int64)&v79,
                    v54,
                    v55,
                    0,
                    *(double *)a5.m128i_i64,
                    *(double *)v12.m128i_i64,
                    *(double *)v13.m128i_i64,
                    *(_OWORD *)a4);
  result = v57;
  a4[2] = v57;
LABEL_49:
  v42 = v79;
  if ( v79 )
    return sub_161E7C0((__int64)&v79, v42);
  return result;
}
