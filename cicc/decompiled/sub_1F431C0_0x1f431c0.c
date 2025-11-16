// Function: sub_1F431C0
// Address: 0x1f431c0
//
void __fastcall sub_1F431C0(unsigned int a1, __int64 *a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  __int64 *v7; // r15
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int8 v11; // r13
  __int64 v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r13
  __int64 (__fastcall *v15)(__int64, __int64, __int64, __int64, __int64); // rax
  int v16; // r14d
  __int64 (__fastcall *v17)(__int64, __int64, __int64, __int64, __int64); // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned __int8 v20; // r9
  unsigned __int8 v21; // r13
  unsigned __int8 v22; // al
  __int64 v23; // rdi
  int v24; // r9d
  __int64 v25; // rcx
  __int64 v26; // rsi
  int v27; // r8d
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // rdi
  __int64 v31; // r11
  int v32; // ebx
  __int8 v33; // r15
  __int8 v34; // r14
  unsigned __int64 v35; // rax
  int v36; // r13d
  __m128i *v37; // rdx
  __int8 v38; // r14
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int64 v42; // r9
  unsigned int v43; // r13d
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int8 v49; // al
  int v50; // r14d
  __int64 v51; // rax
  __int64 v52; // rax
  __int8 v53; // al
  int v54; // eax
  unsigned int v55; // eax
  __int64 v56; // [rsp+0h] [rbp-180h]
  __int64 v57; // [rsp+8h] [rbp-178h]
  const void *v59; // [rsp+20h] [rbp-160h]
  __int64 v60; // [rsp+28h] [rbp-158h]
  unsigned __int64 v61; // [rsp+30h] [rbp-150h]
  __int64 v62; // [rsp+38h] [rbp-148h]
  __int64 v63; // [rsp+40h] [rbp-140h]
  __int64 v65; // [rsp+50h] [rbp-130h]
  __int64 v66; // [rsp+50h] [rbp-130h]
  __int64 v67; // [rsp+50h] [rbp-130h]
  __int64 v68; // [rsp+50h] [rbp-130h]
  unsigned __int8 v69; // [rsp+50h] [rbp-130h]
  __int64 *v70; // [rsp+50h] [rbp-130h]
  __int64 v71; // [rsp+50h] [rbp-130h]
  __int64 v72; // [rsp+50h] [rbp-130h]
  __int64 v73; // [rsp+50h] [rbp-130h]
  __int64 v74; // [rsp+50h] [rbp-130h]
  __int64 v76; // [rsp+60h] [rbp-120h]
  __int64 v77; // [rsp+68h] [rbp-118h] BYREF
  __int8 v78; // [rsp+7Bh] [rbp-105h] BYREF
  int v79; // [rsp+7Ch] [rbp-104h] BYREF
  __m128i v80; // [rsp+80h] [rbp-100h] BYREF
  __int64 v81; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v82; // [rsp+98h] [rbp-E8h]
  __int64 v83; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v84; // [rsp+A8h] [rbp-D8h]
  __int64 v85; // [rsp+B0h] [rbp-D0h] BYREF
  __int64 v86; // [rsp+B8h] [rbp-C8h]
  __int64 v87; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v88; // [rsp+C8h] [rbp-B8h]
  __m128i v89; // [rsp+D0h] [rbp-B0h] BYREF
  __m128i v90; // [rsp+E0h] [rbp-A0h] BYREF
  __m128i v91; // [rsp+F0h] [rbp-90h] BYREF
  _BYTE *v92; // [rsp+100h] [rbp-80h] BYREF
  __int64 v93; // [rsp+108h] [rbp-78h]
  _BYTE v94[112]; // [rsp+110h] [rbp-70h] BYREF

  v77 = a3;
  v92 = v94;
  v93 = 0x400000000LL;
  sub_20C7CE0(a5, a6, a2, &v92, 0, 0);
  if ( (_DWORD)v93 )
  {
    v7 = &v77;
    v76 = 0;
    v60 = 16LL * (unsigned int)v93;
    v59 = (const void *)(a4 + 16);
    while ( 1 )
    {
      v80 = _mm_loadu_si128((const __m128i *)&v92[v76]);
      if ( !(unsigned __int8)sub_1560260(v7, 0, 40) && !(unsigned __int8)sub_1560260(v7, 0, 58) )
        goto LABEL_6;
      v11 = v80.m128i_i8[0];
      if ( v80.m128i_i8[0] )
      {
        v12 = (unsigned int)v80.m128i_u8[0] - 14;
        LOBYTE(v12) = (unsigned __int8)(v80.m128i_i8[0] - 14) <= 0x47u;
        if ( !((unsigned __int8)v12 | ((unsigned __int8)(v80.m128i_i8[0] - 2) <= 5u)) )
          goto LABEL_6;
      }
      else if ( !(unsigned __int8)sub_1F58CF0(&v80) )
      {
        goto LABEL_6;
      }
      v89.m128i_i64[1] = 0;
      v38 = a5[1160];
      v89.m128i_i8[0] = v38;
      if ( v11 == v38 )
        break;
      if ( !v11 )
        goto LABEL_36;
      v43 = sub_1F3E310(&v80);
      if ( !v38 )
      {
LABEL_37:
        v44 = sub_1F58D40(&v89, 0, v39, v40, v41, v42);
        goto LABEL_53;
      }
LABEL_52:
      v44 = sub_1F3E310(&v89);
LABEL_53:
      if ( v44 > v43 )
      {
        v80.m128i_i8[0] = v38;
        v80.m128i_i64[1] = 0;
      }
LABEL_6:
      v13 = *(_QWORD *)a5;
      v14 = *a2;
      v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a5 + 392LL);
      if ( v15 != sub_1F42F80 )
      {
        v54 = v15((__int64)a5, *a2, a1, v80.m128i_u32[0], v80.m128i_i64[1]);
        v13 = *(_QWORD *)a5;
        v16 = v54;
        v14 = *a2;
        goto LABEL_9;
      }
      LOBYTE(v81) = v80.m128i_i8[0];
      v82 = v80.m128i_i64[1];
      if ( v80.m128i_i8[0] )
      {
        v16 = (unsigned __int8)a5[v80.m128i_u8[0] + 1040];
        goto LABEL_9;
      }
      v71 = v80.m128i_i64[1];
      if ( !(unsigned __int8)sub_1F58D20(&v81) )
      {
        v50 = sub_1F58D40(&v81, a5, v45, v46, v47, v48);
        v83 = v81;
        v63 = v81;
        v84 = v82;
        v72 = v82;
        if ( (unsigned __int8)sub_1F58D20(&v83) )
        {
          v89.m128i_i8[0] = 0;
          v89.m128i_i64[1] = 0;
          LOBYTE(v85) = 0;
          sub_1F426C0((__int64)a5, v14, (unsigned int)v83, v84, (__int64)&v89, (unsigned int *)&v87, &v85);
          v53 = v85;
          goto LABEL_59;
        }
        sub_1F40D10((__int64)&v89, (__int64)a5, v14, v63, v72);
        v51 = v89.m128i_u8[8];
        LOBYTE(v85) = v89.m128i_i8[8];
        v86 = v90.m128i_i64[0];
        if ( v89.m128i_i8[8] )
          goto LABEL_58;
        v73 = v90.m128i_i64[0];
        if ( (unsigned __int8)sub_1F58D20(&v85) )
        {
          v89.m128i_i8[0] = 0;
          v89.m128i_i64[1] = 0;
          LOBYTE(v79) = 0;
          sub_1F426C0((__int64)a5, v14, (unsigned int)v85, v73, (__int64)&v89, (unsigned int *)&v87, &v79);
          v53 = v79;
          goto LABEL_59;
        }
        sub_1F40D10((__int64)&v89, (__int64)a5, v14, v85, v86);
        v51 = v89.m128i_u8[8];
        LOBYTE(v87) = v89.m128i_i8[8];
        v88 = v90.m128i_i64[0];
        if ( v89.m128i_i8[8] )
        {
LABEL_58:
          v53 = a5[v51 + 1155];
          goto LABEL_59;
        }
        v74 = v90.m128i_i64[0];
        if ( (unsigned __int8)sub_1F58D20(&v87) )
        {
          v89.m128i_i8[0] = 0;
          v89.m128i_i64[1] = 0;
          v78 = 0;
          sub_1F426C0((__int64)a5, v14, (unsigned int)v87, v74, (__int64)&v89, (unsigned int *)&v79, &v78);
          v53 = v78;
        }
        else
        {
          sub_1F40D10((__int64)&v89, (__int64)a5, v14, v87, v88);
          v52 = v56;
          LOBYTE(v52) = v89.m128i_i8[8];
          v56 = v52;
          v53 = sub_1D5E9F0((__int64)a5, v14, (unsigned int)v52, v90.m128i_i64[0]);
        }
LABEL_59:
        v89.m128i_i8[0] = v53;
        v55 = sub_1F3E310(&v89);
        v16 = (v55 + v50 - 1) / v55;
        v13 = *(_QWORD *)a5;
        v14 = *a2;
LABEL_9:
        v17 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(v13 + 384);
        if ( v17 != sub_1F42DB0 )
          goto LABEL_40;
        goto LABEL_10;
      }
      v89.m128i_i8[0] = 0;
      v89.m128i_i64[1] = 0;
      LOBYTE(v85) = 0;
      v16 = sub_1F426C0((__int64)a5, v14, (unsigned int)v81, v71, (__int64)&v89, (unsigned int *)&v87, &v85);
      v14 = *a2;
      v17 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, __int64))(*(_QWORD *)a5 + 384LL);
      if ( v17 != sub_1F42DB0 )
      {
LABEL_40:
        v20 = v17((__int64)a5, v14, a1, v80.m128i_u32[0], v80.m128i_i64[1]);
        goto LABEL_20;
      }
LABEL_10:
      v18 = v80.m128i_u8[0];
      LOBYTE(v81) = v80.m128i_i8[0];
      v82 = v80.m128i_i64[1];
      if ( !v80.m128i_i8[0] )
      {
        v65 = v80.m128i_i64[1];
        if ( (unsigned __int8)sub_1F58D20(&v81) )
        {
          v89.m128i_i8[0] = 0;
          v89.m128i_i64[1] = 0;
          LOBYTE(v85) = 0;
          sub_1F426C0((__int64)a5, v14, (unsigned int)v81, v65, (__int64)&v89, (unsigned int *)&v87, &v85);
LABEL_56:
          v20 = v85;
          goto LABEL_20;
        }
        sub_1F40D10((__int64)&v89, (__int64)a5, v14, v81, v82);
        v18 = v89.m128i_u8[8];
        LOBYTE(v83) = v89.m128i_i8[8];
        v84 = v90.m128i_i64[0];
        if ( !v89.m128i_i8[8] )
        {
          v66 = v90.m128i_i64[0];
          if ( (unsigned __int8)sub_1F58D20(&v83) )
          {
            v89.m128i_i8[0] = 0;
            v89.m128i_i64[1] = 0;
            LOBYTE(v85) = 0;
            sub_1F426C0((__int64)a5, v14, (unsigned int)v83, v66, (__int64)&v89, (unsigned int *)&v87, &v85);
            goto LABEL_56;
          }
          sub_1F40D10((__int64)&v89, (__int64)a5, v14, v83, v84);
          v18 = v89.m128i_u8[8];
          LOBYTE(v85) = v89.m128i_i8[8];
          v86 = v90.m128i_i64[0];
          if ( !v89.m128i_i8[8] )
          {
            v67 = v90.m128i_i64[0];
            if ( (unsigned __int8)sub_1F58D20(&v85) )
            {
              v89.m128i_i8[0] = 0;
              v89.m128i_i64[1] = 0;
              LOBYTE(v79) = 0;
              sub_1F426C0((__int64)a5, v14, (unsigned int)v85, v67, (__int64)&v89, (unsigned int *)&v87, &v79);
              v20 = v79;
              goto LABEL_20;
            }
            sub_1F40D10((__int64)&v89, (__int64)a5, v14, v85, v86);
            v18 = v89.m128i_u8[8];
            LOBYTE(v87) = v89.m128i_i8[8];
            v88 = v90.m128i_i64[0];
            if ( !v89.m128i_i8[8] )
            {
              v68 = v90.m128i_i64[0];
              if ( (unsigned __int8)sub_1F58D20(&v87) )
              {
                v89.m128i_i8[0] = 0;
                v89.m128i_i64[1] = 0;
                v78 = 0;
                sub_1F426C0((__int64)a5, v14, (unsigned int)v87, v68, (__int64)&v89, (unsigned int *)&v79, &v78);
                v20 = v78;
              }
              else
              {
                sub_1F40D10((__int64)&v89, (__int64)a5, v14, v87, v88);
                v19 = v57;
                LOBYTE(v19) = v89.m128i_i8[8];
                v57 = v19;
                v20 = sub_1D5E9F0((__int64)a5, v14, (unsigned int)v19, v90.m128i_i64[0]);
              }
              goto LABEL_20;
            }
          }
        }
      }
      v20 = a5[v18 + 1155];
LABEL_20:
      v69 = v20;
      v21 = sub_1560260(v7, 0, 12);
      v22 = sub_1560260(v7, 0, 40);
      v23 = 0;
      v24 = v69;
      v25 = v22;
      if ( !v22 )
      {
        v49 = sub_1560260(v7, 0, 58);
        v25 = 0;
        v24 = v69;
        v23 = v49;
      }
      if ( v16 )
      {
        v26 = v21;
        v27 = 0;
        v28 = v23 | (2 * v25);
        v29 = *(unsigned int *)(a4 + 8);
        v30 = a4;
        v31 = v80.m128i_i64[1];
        v70 = v7;
        v32 = v16;
        v33 = v80.m128i_i8[0];
        v34 = v24;
        v35 = ((4 * v26) | v28) & 0xFFFFFFFF07FFFFFFLL;
        v36 = 0;
        do
        {
          v91.m128i_i8[0] = 1;
          *(__int64 *)((char *)v91.m128i_i64 + 4) = 0;
          v89.m128i_i8[8] = v34;
          v90.m128i_i8[0] = v33;
          v89.m128i_i64[0] = v35 | v89.m128i_i32[0] & 0xF8000000;
          v90.m128i_i64[1] = v31;
          if ( (unsigned int)v29 >= *(_DWORD *)(v30 + 12) )
          {
            v61 = v35;
            v62 = v31;
            sub_16CD150(v30, v59, 0, 48, v27, v24);
            v35 = v61;
            v31 = v62;
            v29 = *(unsigned int *)(v30 + 8);
          }
          ++v36;
          v37 = (__m128i *)(*(_QWORD *)v30 + 48 * v29);
          *v37 = _mm_loadu_si128(&v89);
          v37[1] = _mm_loadu_si128(&v90);
          v37[2] = _mm_loadu_si128(&v91);
          v29 = (unsigned int)(*(_DWORD *)(v30 + 8) + 1);
          *(_DWORD *)(v30 + 8) = v29;
        }
        while ( v32 != v36 );
        v7 = v70;
        a4 = v30;
      }
      v76 += 16;
      if ( v76 == v60 )
        goto LABEL_29;
    }
    if ( v38 || !v80.m128i_i64[1] )
      goto LABEL_6;
LABEL_36:
    v43 = sub_1F58D40(&v80, 0, v12, v8, v9, v10);
    if ( !v38 )
      goto LABEL_37;
    goto LABEL_52;
  }
LABEL_29:
  if ( v92 != v94 )
    _libc_free((unsigned __int64)v92);
}
