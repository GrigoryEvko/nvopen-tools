// Function: sub_3846C20
// Address: 0x3846c20
//
void __fastcall sub_3846C20(__int64 a1, unsigned __int64 a2, __m128i *a3, __int64 a4)
{
  __int16 *v7; // rax
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned __int16 v10; // di
  __int64 v11; // r8
  __int64 v12; // r10
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int128 v17; // xmm1
  __int64 v18; // r14
  __int64 v19; // rdi
  __int64 v20; // rax
  bool v21; // cc
  unsigned __int64 v22; // rax
  __m128i v23; // xmm0
  __int64 v24; // r15
  __int64 (__fastcall *v25)(__int64, __int64, unsigned int); // rax
  int v26; // edx
  unsigned __int16 v27; // ax
  unsigned __int8 *v28; // rax
  __int64 v29; // r10
  __int64 v30; // rdx
  __int64 v31; // r8
  __int64 v32; // r9
  char v33; // r15
  __int64 v34; // rax
  __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 v37; // r8
  unsigned int v38; // ecx
  _QWORD *v39; // r10
  unsigned __int8 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rsi
  __int32 v45; // edx
  __int64 v46; // rdx
  __int64 v47; // r14
  __int64 v48; // rax
  __int64 v49; // rdx
  unsigned int v50; // eax
  __int64 v51; // r9
  unsigned int v52; // ecx
  __int64 v53; // r8
  unsigned __int8 *v54; // rax
  __int64 v55; // rsi
  int v56; // edx
  unsigned __int64 v57; // r14
  __m128i v58; // xmm0
  __int64 v59; // rdx
  __int64 v60; // rcx
  _QWORD *v61; // rax
  unsigned __int8 *v62; // rax
  unsigned int v63; // esi
  __int32 v64; // edx
  __int32 v65; // edi
  __int64 v66; // rdx
  int v67; // edx
  __int64 v68; // rdx
  __int64 v69; // [rsp+0h] [rbp-230h]
  unsigned int v70; // [rsp+8h] [rbp-228h]
  _QWORD *v71; // [rsp+10h] [rbp-220h]
  __int64 v72; // [rsp+10h] [rbp-220h]
  char v73; // [rsp+18h] [rbp-218h]
  __int64 v74; // [rsp+18h] [rbp-218h]
  __int64 v75; // [rsp+30h] [rbp-200h]
  __int64 v76; // [rsp+30h] [rbp-200h]
  __int64 v77; // [rsp+38h] [rbp-1F8h]
  const __m128i *v78; // [rsp+38h] [rbp-1F8h]
  __int64 v79; // [rsp+40h] [rbp-1F0h]
  unsigned __int16 v80; // [rsp+4Eh] [rbp-1E2h]
  __int64 v81; // [rsp+68h] [rbp-1C8h]
  __int64 v83; // [rsp+78h] [rbp-1B8h]
  unsigned int v84; // [rsp+78h] [rbp-1B8h]
  __int32 v85; // [rsp+C8h] [rbp-168h]
  __m128i v86; // [rsp+D0h] [rbp-160h] BYREF
  __int64 v87; // [rsp+E0h] [rbp-150h] BYREF
  int v88; // [rsp+E8h] [rbp-148h]
  unsigned __int64 v89; // [rsp+F0h] [rbp-140h] BYREF
  char v90; // [rsp+F8h] [rbp-138h]
  __int64 v91; // [rsp+100h] [rbp-130h] BYREF
  __int64 v92; // [rsp+108h] [rbp-128h]
  unsigned __int64 v93[2]; // [rsp+110h] [rbp-120h] BYREF
  _OWORD v94[4]; // [rsp+120h] [rbp-110h] BYREF
  unsigned __int64 v95[2]; // [rsp+160h] [rbp-D0h] BYREF
  _OWORD v96[4]; // [rsp+170h] [rbp-C0h] BYREF
  __m128i *v97; // [rsp+1B0h] [rbp-80h] BYREF
  __int64 v98; // [rsp+1B8h] [rbp-78h]
  __m128i v99[7]; // [rsp+1C0h] [rbp-70h] BYREF

  v7 = *(__int16 **)(a2 + 48);
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(a1 + 8);
  v10 = *v7;
  v11 = *((_QWORD *)v7 + 1);
  v12 = *(_QWORD *)(v9 + 64);
  v80 = *v7;
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v8 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v97, v8, v12, v10, v11);
    v86.m128i_i16[0] = v98;
    v86.m128i_i64[1] = v99[0].m128i_i64[0];
  }
  else
  {
    v86.m128i_i32[0] = v13(v8, v12, v80, v11);
    v86.m128i_i64[1] = v68;
  }
  v14 = *(_QWORD *)(a2 + 40);
  v15 = *(_QWORD *)(a2 + 80);
  v16 = *(_QWORD *)(v14 + 8);
  v17 = (__int128)_mm_loadu_si128((const __m128i *)(v14 + 40));
  v18 = *(_QWORD *)v14;
  v87 = v15;
  v81 = v16;
  if ( v15 )
    sub_B96E90((__int64)&v87, v15, 1);
  v19 = *(_QWORD *)(a1 + 8);
  v88 = *(_DWORD *)(a2 + 72);
  if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)v19 + 544LL) - 42) > 1 )
  {
    v59 = *(_QWORD *)(a2 + 40);
    v60 = *(_QWORD *)(*(_QWORD *)(v59 + 120) + 96LL);
    v61 = *(_QWORD **)(v60 + 24);
    if ( *(_DWORD *)(v60 + 32) > 0x40u )
      v61 = (_QWORD *)*v61;
    v62 = sub_3411830(
            (__int64 *)v19,
            v86.m128i_u32[0],
            v86.m128i_i64[1],
            (__int64)&v87,
            v18,
            v81,
            v17,
            *(_OWORD *)(v59 + 80),
            (unsigned int)v61);
    v63 = v86.m128i_i32[0];
    v65 = v64;
    v66 = v86.m128i_i64[1];
    a3->m128i_i64[0] = (__int64)v62;
    a3->m128i_i32[2] = v65;
    *(_QWORD *)a4 = sub_3411830(
                      *(__int64 **)(a1 + 8),
                      v63,
                      v66,
                      (__int64)&v87,
                      (__int64)v62,
                      1,
                      v17,
                      *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL),
                      0);
    *(_DWORD *)(a4 + 8) = v67;
  }
  else
  {
    v79 = *(_QWORD *)(v19 + 16);
    v83 = sub_2E79000(*(__int64 **)(v19 + 40));
    v20 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL) + 96LL);
    v21 = *(_DWORD *)(v20 + 32) <= 0x40u;
    v22 = *(_QWORD *)(v20 + 24);
    if ( !v21 )
      v22 = *(_QWORD *)v22;
    v23 = _mm_load_si128(&v86);
    v24 = *(_QWORD *)(a1 + 8);
    v95[0] = (unsigned __int64)v96;
    v94[0] = v23;
    v96[0] = v23;
    v93[0] = (unsigned __int64)v94;
    v93[1] = 0x400000001LL;
    v95[1] = 0x400000001LL;
    v77 = (v22 | 8) & -(__int64)(v22 | 8);
    v25 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v79 + 32LL);
    if ( v25 == sub_2D42F30 )
    {
      v26 = sub_AE2980(v83, 0)[1];
      v27 = 2;
      if ( v26 != 1 )
      {
        v27 = 3;
        if ( v26 != 2 )
        {
          v27 = 4;
          if ( v26 != 4 )
          {
            v27 = 5;
            if ( v26 != 8 )
            {
              v27 = 6;
              if ( v26 != 16 )
              {
                v27 = 7;
                if ( v26 != 32 )
                {
                  v27 = 8;
                  if ( v26 != 64 )
                    v27 = 9 * (v26 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v27 = v25(v79, v83, 0);
    }
    v28 = sub_3400BD0(v24, 0, (__int64)&v87, v27, 0, 1u, v23, 0);
    v29 = *(_QWORD *)(a1 + 8);
    v99[0].m128i_i64[0] = (__int64)v28;
    v99[0].m128i_i64[1] = (unsigned int)v30;
    v97 = v99;
    v98 = 0x400000001LL;
    v71 = (_QWORD *)v29;
    v75 = sub_3007410((__int64)&v86, *(__int64 **)(v29 + 64), v30, (__int64)v99, v31, v32);
    v33 = sub_AE5020(v83, v75);
    v34 = sub_9208B0(v83, v75);
    v92 = v35;
    v91 = v34;
    v89 = ((1LL << v33) + ((unsigned __int64)(v34 + 7) >> 3) - 1) >> v33 << v33;
    v90 = v35;
    v36 = sub_CA1930(&v89);
    v37 = *(_QWORD *)(a2 + 40);
    v38 = v36;
    v39 = v71;
    v91 = v87;
    if ( v87 )
    {
      v69 = v37;
      v70 = v36;
      sub_B96E90((__int64)&v91, v87, 1);
      v37 = v69;
      v38 = v70;
      v39 = v71;
    }
    LODWORD(v92) = v88;
    v40 = sub_34118E0(v39, (__int64)v93, (__int64)&v91, v18, v81, v77, v23, v17, *(_OWORD *)(v37 + 80), v38, v99);
    v44 = v91;
    v85 = v45;
    v46 = (__int64)a3;
    a3->m128i_i64[0] = (__int64)v40;
    a3->m128i_i32[2] = v85;
    if ( v44 )
      sub_B91220((__int64)&v91, v44);
    v47 = *(_QWORD *)(a1 + 8);
    v78 = v97;
    v72 = sub_3007410((__int64)&v86, *(__int64 **)(v47 + 64), v46, v41, v42, v43);
    v73 = sub_AE5020(v83, v72);
    v48 = sub_9208B0(v83, v72);
    v92 = v49;
    v91 = v48;
    v89 = (((unsigned __int64)(v48 + 7) >> 3) + (1LL << v73) - 1) >> v73 << v73;
    v90 = v49;
    v50 = sub_CA1930(&v89);
    v51 = *(_QWORD *)(a2 + 40);
    v52 = v50;
    v91 = v87;
    v53 = a3->m128i_i64[0];
    if ( v87 )
    {
      v74 = a3->m128i_i64[0];
      v76 = v51;
      v84 = v50;
      sub_B96E90((__int64)&v91, v87, 1);
      v53 = v74;
      v51 = v76;
      v52 = v84;
    }
    LODWORD(v92) = v88;
    v54 = sub_34118E0((_QWORD *)v47, (__int64)v95, (__int64)&v91, v53, 1, 0, v23, v17, *(_OWORD *)(v51 + 80), v52, v78);
    v55 = v91;
    *(_QWORD *)a4 = v54;
    *(_DWORD *)(a4 + 8) = v56;
    if ( v55 )
      sub_B91220((__int64)&v91, v55);
    if ( v97 != v99 )
      _libc_free((unsigned __int64)v97);
    if ( (_OWORD *)v95[0] != v96 )
      _libc_free(v95[0]);
    if ( (_OWORD *)v93[0] != v94 )
      _libc_free(v93[0]);
  }
  v57 = *(_QWORD *)a4;
  if ( *(_BYTE *)sub_2E79000(*(__int64 **)(*(_QWORD *)(a1 + 8) + 40LL)) == 1 || v80 == 16 )
  {
    v58 = _mm_loadu_si128(a3);
    a3->m128i_i64[0] = *(_QWORD *)a4;
    a3->m128i_i32[2] = *(_DWORD *)(a4 + 8);
    *(_QWORD *)a4 = v58.m128i_i64[0];
    *(_DWORD *)(a4 + 8) = v58.m128i_i32[2];
  }
  sub_3760E70(a1, a2, 1, v57, v81 & 0xFFFFFFFF00000000LL | 1);
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
}
