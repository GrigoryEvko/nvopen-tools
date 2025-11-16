// Function: sub_3324F40
// Address: 0x3324f40
//
__int64 __fastcall sub_3324F40(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v9; // r9d
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rcx
  __m128i v14; // xmm0
  __int64 v15; // r13
  __int64 v16; // r11
  unsigned __int16 *v17; // rax
  unsigned __int16 v18; // r8
  __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // rdi
  int v22; // eax
  char v23; // al
  unsigned __int16 v24; // r8
  int v25; // eax
  char v26; // al
  int v27; // r9d
  __int64 v28; // rax
  char v29; // cl
  unsigned int v30; // eax
  unsigned int v31; // esi
  unsigned int v32; // r11d
  int v33; // r9d
  __int64 v34; // rcx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  __int64 v39; // rdx
  __int64 v40; // r13
  __int64 v41; // r12
  __int128 v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // r12
  __int64 v47; // rdx
  unsigned __int64 v48; // r13
  __int128 v49; // rax
  __int64 v50; // rax
  __int64 v51; // rdi
  __int64 v52; // rdx
  int v53; // r9d
  __int64 v54; // rax
  __int64 v55; // rdi
  __int64 v56; // rdx
  int v57; // r9d
  __m128i v58; // xmm1
  __int64 v59; // rdx
  __int128 v60; // [rsp-20h] [rbp-110h]
  __int128 v61; // [rsp-20h] [rbp-110h]
  __int128 v62; // [rsp-20h] [rbp-110h]
  __int128 v63; // [rsp-20h] [rbp-110h]
  __int128 v64; // [rsp-10h] [rbp-100h]
  __int128 v65; // [rsp-10h] [rbp-100h]
  unsigned int v66; // [rsp+10h] [rbp-E0h]
  int v67; // [rsp+20h] [rbp-D0h]
  unsigned __int16 v68; // [rsp+28h] [rbp-C8h]
  unsigned int v69; // [rsp+28h] [rbp-C8h]
  unsigned int v70; // [rsp+28h] [rbp-C8h]
  __int64 v71; // [rsp+30h] [rbp-C0h]
  __int64 v72; // [rsp+38h] [rbp-B8h]
  unsigned __int16 v73; // [rsp+38h] [rbp-B8h]
  __int64 v74; // [rsp+38h] [rbp-B8h]
  __m128i v75; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v76; // [rsp+50h] [rbp-A0h]
  __int64 v77; // [rsp+58h] [rbp-98h]
  __int64 v78; // [rsp+60h] [rbp-90h]
  __int64 v79; // [rsp+68h] [rbp-88h]
  __int64 v80; // [rsp+70h] [rbp-80h]
  __int64 v81; // [rsp+78h] [rbp-78h]
  __int64 v82; // [rsp+80h] [rbp-70h]
  __int64 v83; // [rsp+88h] [rbp-68h]
  __int64 v84; // [rsp+90h] [rbp-60h] BYREF
  int v85; // [rsp+98h] [rbp-58h]
  __int64 v86; // [rsp+A0h] [rbp-50h] BYREF
  unsigned __int64 v87; // [rsp+A8h] [rbp-48h]
  __m128i v88; // [rsp+B0h] [rbp-40h]

  result = sub_3324380((__int64)a1, a2, 173, a4, a5, a6);
  if ( !result )
  {
    v10 = *(__int64 **)(a2 + 40);
    v11 = *(_QWORD *)(a2 + 80);
    v12 = *v10;
    v13 = *v10;
    v14 = _mm_loadu_si128((const __m128i *)(v10 + 5));
    v15 = v10[1];
    v16 = v10[5];
    v17 = *(unsigned __int16 **)(a2 + 48);
    v75 = v14;
    v18 = *v17;
    v19 = *((_QWORD *)v17 + 1);
    v84 = v11;
    v67 = v19;
    v20 = v18;
    if ( v11 )
    {
      v68 = v18;
      v71 = v16;
      v72 = v13;
      sub_B96E90((__int64)&v84, v11, 1);
      v18 = v68;
      v16 = v71;
      v13 = v72;
    }
    v21 = *a1;
    v85 = *(_DWORD *)(a2 + 72);
    v22 = *(_DWORD *)(v13 + 24);
    if ( v22 == 35 || v22 == 11 )
    {
      v25 = *(_DWORD *)(v16 + 24);
      if ( v25 == 11 || v25 == 35 )
      {
        *((_QWORD *)&v60 + 1) = v15;
        *(_QWORD *)&v60 = v12;
        result = sub_3411F20(
                   v21,
                   63,
                   (unsigned int)&v84,
                   *(_QWORD *)(a2 + 48),
                   *(_DWORD *)(a2 + 68),
                   v9,
                   v60,
                   *(_OWORD *)&v75);
        goto LABEL_11;
      }
    }
    v73 = v18;
    v23 = sub_33E2390(v21, v12, v15, 1);
    v24 = v73;
    if ( v23 )
    {
      v26 = sub_33E2390(*a1, v75.m128i_i64[0], v75.m128i_i64[1], 1);
      v24 = v73;
      if ( !v26 )
      {
        *((_QWORD *)&v64 + 1) = v15;
        *(_QWORD *)&v64 = v12;
        result = sub_3411F20(
                   *a1,
                   63,
                   (unsigned int)&v84,
                   *(_QWORD *)(a2 + 48),
                   *(_DWORD *)(a2 + 68),
                   v27,
                   *(_OWORD *)&v75,
                   v64);
        goto LABEL_11;
      }
    }
    if ( !v24 || (unsigned __int16)(v24 - 17) <= 0xD3u )
    {
LABEL_10:
      result = 0;
      goto LABEL_11;
    }
    if ( v24 == 1 || (unsigned __int16)(v24 - 504) <= 7u )
      BUG();
    v28 = 16LL * (v24 - 1);
    v29 = byte_444C4A0[v28 + 8];
    v86 = *(_QWORD *)&byte_444C4A0[v28];
    LOBYTE(v87) = v29;
    v30 = sub_CA1930(&v86);
    v31 = 2 * v30;
    v32 = v30;
    if ( 2 * v30 == 2 )
    {
      v36 = a1[1];
      v33 = 3;
      v34 = 3;
      v74 = 0;
    }
    else
    {
      switch ( v31 )
      {
        case 4u:
          v36 = a1[1];
          v33 = 4;
          v34 = 4;
          v74 = 0;
          break;
        case 8u:
          v36 = a1[1];
          v33 = 5;
          v34 = 5;
          v74 = 0;
          break;
        case 0x10u:
          v36 = a1[1];
          v33 = 6;
          v34 = 6;
          v74 = 0;
          break;
        case 0x20u:
          v36 = a1[1];
          v33 = 7;
          v34 = 7;
          v74 = 0;
          break;
        case 0x40u:
          v36 = a1[1];
          v33 = 8;
          v34 = 8;
          v74 = 0;
          break;
        case 0x80u:
          v36 = a1[1];
          v33 = 9;
          v34 = 9;
          v74 = 0;
          break;
        default:
          v69 = v30;
          LODWORD(v37) = sub_3007020(*(_QWORD **)(*a1 + 64LL), v31);
          v32 = v69;
          v33 = v37;
          v34 = (unsigned __int16)v37;
          v74 = v35;
          v36 = a1[1];
          v37 = (unsigned __int16)v37;
          if ( (_WORD)v37 == 1 )
          {
LABEL_30:
            v70 = v32;
            if ( !*(_BYTE *)(v36 + 500 * v37 + 6472) )
            {
              *((_QWORD *)&v65 + 1) = v15;
              *(_QWORD *)&v65 = v12;
              v66 = v33;
              v38 = sub_33FAF80(*a1, 213, (unsigned int)&v84, v33, v74, v33, v65);
              v40 = v39;
              v41 = v38;
              *(_QWORD *)&v42 = sub_33FAF80(*a1, 213, (unsigned int)&v84, v66, v74, v66, *(_OWORD *)&v75);
              v43 = *a1;
              *((_QWORD *)&v61 + 1) = v40;
              *(_QWORD *)&v61 = v41;
              v75.m128i_i64[1] = *((_QWORD *)&v42 + 1);
              v44 = sub_3406EB0(v43, 58, (unsigned int)&v84, v66, v74, v66, v61, v42);
              v45 = *a1;
              v46 = v44;
              v82 = v44;
              v83 = v47;
              v48 = (unsigned int)v47 | v40 & 0xFFFFFFFF00000000LL;
              *(_QWORD *)&v49 = sub_3400E40(v45, v70, v66, v74, &v84);
              *((_QWORD *)&v62 + 1) = v48;
              *(_QWORD *)&v62 = v46;
              v50 = sub_3406EB0(v45, 192, (unsigned int)&v84, v66, v74, v66, v62, v49);
              v51 = *a1;
              v75.m128i_i64[0] = v50;
              v80 = v50;
              v81 = v52;
              v75.m128i_i64[1] = (unsigned int)v52 | v75.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              v54 = sub_33FAF80(v51, 216, (unsigned int)&v84, v20, v67, v53, __PAIR128__(v75.m128i_u64[1], v50));
              *((_QWORD *)&v63 + 1) = v48;
              v55 = *a1;
              v78 = v54;
              v75.m128i_i64[0] = v54;
              v79 = v56;
              *(_QWORD *)&v63 = v46;
              v75.m128i_i64[1] = (unsigned int)v56 | v75.m128i_i64[1] & 0xFFFFFFFF00000000LL;
              v76 = sub_33FAF80(v55, 216, (unsigned int)&v84, v20, v67, v57, v63);
              v58 = _mm_load_si128(&v75);
              v77 = v59;
              v86 = v76;
              v88 = v58;
              v87 = (unsigned int)v59 | v48 & 0xFFFFFFFF00000000LL;
              result = sub_32EB790((__int64)a1, a2, &v86, 2, 1);
LABEL_11:
              if ( v84 )
              {
                v75.m128i_i64[0] = result;
                sub_B91220((__int64)&v84, v84);
                return v75.m128i_i64[0];
              }
              return result;
            }
            goto LABEL_10;
          }
          if ( !(_WORD)v33 )
            goto LABEL_10;
          break;
      }
    }
    v37 = (unsigned __int16)v34;
    if ( !*(_QWORD *)(v36 + 8 * v34 + 112) )
      goto LABEL_10;
    goto LABEL_30;
  }
  return result;
}
