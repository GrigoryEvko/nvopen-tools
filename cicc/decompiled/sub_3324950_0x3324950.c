// Function: sub_3324950
// Address: 0x3324950
//
__int64 __fastcall sub_3324950(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  int v9; // r9d
  __int64 *v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rbx
  __m128i v13; // xmm0
  __int64 v14; // r12
  __int64 v15; // r13
  __int64 v16; // rcx
  unsigned __int16 *v17; // rax
  unsigned __int16 v18; // di
  __int64 v19; // rax
  int v20; // ebx
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rax
  __int32 v24; // edx
  __int64 v25; // rdx
  int v26; // r9d
  int v27; // eax
  __int64 v28; // rax
  __int8 v29; // cl
  __int64 v30; // rax
  unsigned int v31; // r9d
  __int64 v32; // rdx
  __int64 v33; // r11
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rdi
  __m128i v37; // rax
  __int64 v38; // rdi
  __int128 v39; // rax
  __int64 v40; // rdi
  __int64 v41; // rax
  __int64 v42; // r12
  __int64 v43; // rdx
  __int128 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdi
  __int64 v47; // rdx
  int v48; // r9d
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rdx
  int v52; // r9d
  __int64 v53; // rdx
  __m128i v54; // xmm2
  __m128i v55; // xmm1
  __int128 v56; // [rsp-20h] [rbp-110h]
  __int128 v57; // [rsp-20h] [rbp-110h]
  __int128 v58; // [rsp-10h] [rbp-100h]
  unsigned int v59; // [rsp+8h] [rbp-E8h]
  __int64 *v60; // [rsp+10h] [rbp-E0h]
  unsigned int v61; // [rsp+18h] [rbp-D8h]
  unsigned __int16 v62; // [rsp+20h] [rbp-D0h]
  int v63; // [rsp+28h] [rbp-C8h]
  __m128i v64; // [rsp+30h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v66; // [rsp+50h] [rbp-A0h]
  __int64 v67; // [rsp+58h] [rbp-98h]
  __int64 v68; // [rsp+60h] [rbp-90h]
  __int64 v69; // [rsp+68h] [rbp-88h]
  __int64 v70; // [rsp+70h] [rbp-80h]
  __int64 v71; // [rsp+78h] [rbp-78h]
  __int64 v72; // [rsp+80h] [rbp-70h]
  __int64 v73; // [rsp+88h] [rbp-68h]
  __int64 v74; // [rsp+90h] [rbp-60h] BYREF
  int v75; // [rsp+98h] [rbp-58h]
  __m128i v76; // [rsp+A0h] [rbp-50h] BYREF
  __m128i v77; // [rsp+B0h] [rbp-40h]

  result = sub_3324380((__int64)a1, a2, 172, a4, a5, a6);
  if ( !result )
  {
    v10 = *(__int64 **)(a2 + 40);
    v11 = *(_QWORD *)(a2 + 80);
    v12 = *v10;
    v13 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
    v14 = v10[5];
    v15 = v10[6];
    v16 = v14;
    v17 = *(unsigned __int16 **)(a2 + 48);
    v64.m128i_i64[0] = v12;
    v65 = v13;
    v18 = *v17;
    v19 = *((_QWORD *)v17 + 1);
    v74 = v11;
    v62 = v18;
    v20 = v18;
    v63 = v19;
    if ( v11 )
    {
      sub_B96E90((__int64)&v74, v11, 1);
      v16 = v14;
    }
    v21 = *a1;
    v75 = *(_DWORD *)(a2 + 72);
    v22 = *(_DWORD *)(v64.m128i_i64[0] + 24);
    if ( (v22 == 11 || v22 == 35) && ((v27 = *(_DWORD *)(v16 + 24), v27 == 11) || v27 == 35) )
    {
      *((_QWORD *)&v58 + 1) = v15;
      *(_QWORD *)&v58 = v14;
      result = sub_3411F20(
                 v21,
                 64,
                 (unsigned int)&v74,
                 *(_QWORD *)(a2 + 48),
                 *(_DWORD *)(a2 + 68),
                 v9,
                 *(_OWORD *)&v65,
                 v58);
    }
    else if ( !(unsigned __int8)sub_33E2390(v21, v65.m128i_i64[0], v65.m128i_i64[1], 1)
           || (unsigned __int8)sub_33E2390(*a1, v14, v15, 1) )
    {
      if ( (unsigned __int8)sub_33CF170(v14, v15) )
      {
        v23 = sub_3400BD0(*a1, 0, (unsigned int)&v74, v20, v63, 0, 0);
        v76.m128i_i64[0] = v23;
        v76.m128i_i32[2] = v24;
      }
      else
      {
        if ( !(unsigned __int8)sub_33CF4D0(v14, v15) )
        {
          if ( !v62 || (unsigned __int16)(v62 - 17) <= 0xD3u )
            goto LABEL_28;
          if ( v62 == 1 || (unsigned __int16)(v62 - 504) <= 7u )
            BUG();
          v64.m128i_i64[0] = (__int64)&v76;
          v28 = 16LL * (v62 - 1);
          v29 = byte_444C4A0[v28 + 8];
          v30 = *(_QWORD *)&byte_444C4A0[v28];
          v76.m128i_i8[8] = v29;
          v76.m128i_i64[0] = v30;
          v61 = sub_CA1930(&v76);
          v31 = sub_327FC40(*(_QWORD **)(*a1 + 64LL), 2 * v61);
          v33 = v32;
          v34 = 1;
          v35 = a1[1];
          if ( (_WORD)v31 != 1 )
          {
            if ( !(_WORD)v31 )
              goto LABEL_28;
            v34 = (unsigned __int16)v31;
            if ( !*(_QWORD *)(v35 + 8LL * (unsigned __int16)v31 + 112) )
              goto LABEL_28;
          }
          v60 = (__int64 *)v64.m128i_i64[0];
          if ( !*(_BYTE *)(v35 + 500 * v34 + 6472) )
          {
            v36 = *a1;
            v59 = v31;
            v64.m128i_i64[0] = v33;
            v37.m128i_i64[0] = sub_33FAF80(v36, 214, (unsigned int)&v74, v31, v33, v31, *(_OWORD *)&v65);
            *((_QWORD *)&v57 + 1) = v15;
            v38 = *a1;
            *(_QWORD *)&v57 = v14;
            v65 = v37;
            *(_QWORD *)&v39 = sub_33FAF80(v38, 214, (unsigned int)&v74, v59, v64.m128i_i32[0], v59, v57);
            v40 = *a1;
            v64.m128i_i64[1] = *((_QWORD *)&v39 + 1);
            v41 = sub_3406EB0(v40, 58, (unsigned int)&v74, v59, v64.m128i_i32[0], v59, *(_OWORD *)&v65, v39);
            v42 = *a1;
            v72 = v41;
            v73 = v43;
            v65.m128i_i64[0] = v41;
            v65.m128i_i64[1] = (unsigned int)v43 | v65.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v44 = sub_3400E40(v42, v61, v59, v64.m128i_i64[0], &v74);
            v45 = sub_3406EB0(v42, 192, (unsigned int)&v74, v59, v64.m128i_i32[0], v59, *(_OWORD *)&v65, v44);
            v46 = *a1;
            v70 = v45;
            v64.m128i_i64[0] = v45;
            v71 = v47;
            v64.m128i_i64[1] = (unsigned int)v47 | v64.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v49 = sub_33FAF80(v46, 216, (unsigned int)&v74, v20, v63, v48, __PAIR128__(v64.m128i_u64[1], v45));
            v50 = *a1;
            v68 = v49;
            v64.m128i_i64[0] = v49;
            v69 = v51;
            v64.m128i_i64[1] = (unsigned int)v51 | v64.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v65.m128i_i64[0] = sub_33FAF80(v50, 216, (unsigned int)&v74, v20, v63, v52, *(_OWORD *)&v65);
            v66 = v65.m128i_i64[0];
            v67 = v53;
            v54 = _mm_load_si128(&v64);
            v65.m128i_i64[1] = (unsigned int)v53 | v65.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v55 = _mm_load_si128(&v65);
            v77 = v54;
            v76 = v55;
            result = sub_32EB790((__int64)a1, a2, v60, 2, 1);
          }
          else
          {
LABEL_28:
            result = 0;
            v25 = 0;
          }
          goto LABEL_11;
        }
        v23 = sub_3400BD0(*a1, 0, (unsigned int)&v74, v20, v63, 0, 0);
        v76.m128i_i32[2] = v65.m128i_i32[2];
        v76.m128i_i64[0] = v64.m128i_i64[0];
      }
      v77.m128i_i32[2] = v24;
      v77.m128i_i64[0] = v23;
      result = sub_32EB790((__int64)a1, a2, v76.m128i_i64, 2, 1);
    }
    else
    {
      *((_QWORD *)&v56 + 1) = v15;
      *(_QWORD *)&v56 = v14;
      result = sub_3411F20(
                 *a1,
                 64,
                 (unsigned int)&v74,
                 *(_QWORD *)(a2 + 48),
                 *(_DWORD *)(a2 + 68),
                 v26,
                 v56,
                 *(_OWORD *)&v65);
    }
LABEL_11:
    if ( v74 )
    {
      v64.m128i_i64[0] = v25;
      v65.m128i_i64[0] = result;
      sub_B91220((__int64)&v74, v74);
      return v65.m128i_i64[0];
    }
  }
  return result;
}
