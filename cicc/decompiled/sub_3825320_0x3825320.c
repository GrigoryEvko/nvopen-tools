// Function: sub_3825320
// Address: 0x3825320
//
void __fastcall sub_3825320(__int64 *a1, __int64 a2, __int64 a3, __m128i *a4, __m128i a5)
{
  __int64 v5; // r14
  __int64 *v8; // r12
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // rdx
  int v13; // eax
  __int64 v14; // rdx
  __int64 (__fastcall *v15)(__int64, __int64, unsigned int, __int64); // rax
  unsigned __int16 v16; // r9
  __int64 v17; // r8
  __int64 v18; // r10
  char v19; // al
  __int64 v20; // rax
  __int64 *v21; // r15
  __int64 v22; // r12
  __int64 v23; // rcx
  __int16 v24; // r13
  __int64 v25; // rdx
  __int64 (__fastcall *v26)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  _BOOL8 (__fastcall *v29)(__int64, __int64, __int64, int); // rax
  int v30; // eax
  __int64 v31; // rax
  int v32; // r11d
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rcx
  unsigned int v37; // r14d
  __int64 v38; // rax
  __int64 v39; // rbx
  __int64 v40; // rax
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 v46; // rax
  __int64 *v47; // r12
  __m128i v48; // xmm2
  unsigned int *v49; // rax
  __int64 v50; // rdx
  int v51; // edx
  unsigned int v52; // eax
  __int64 v53; // rdx
  unsigned __int8 *v54; // rax
  _WORD *v55; // rsi
  __m128i v56; // xmm0
  __int64 v57; // rdx
  __int64 v58; // r10
  __int128 v59; // [rsp-10h] [rbp-130h]
  int v60; // [rsp+8h] [rbp-118h]
  __int64 v61; // [rsp+10h] [rbp-110h]
  __int64 v62; // [rsp+18h] [rbp-108h]
  unsigned int v63; // [rsp+2Ch] [rbp-F4h]
  char v64; // [rsp+30h] [rbp-F0h]
  __int64 v65; // [rsp+38h] [rbp-E8h]
  unsigned int v66; // [rsp+38h] [rbp-E8h]
  __int64 v67; // [rsp+38h] [rbp-E8h]
  unsigned __int8 *v68; // [rsp+40h] [rbp-E0h]
  __int64 (__fastcall *v69)(__int64, __int64, unsigned int); // [rsp+50h] [rbp-D0h]
  unsigned int v70; // [rsp+50h] [rbp-D0h]
  int v71; // [rsp+58h] [rbp-C8h]
  __int64 v72; // [rsp+58h] [rbp-C8h]
  int v73; // [rsp+58h] [rbp-C8h]
  unsigned __int8 *v74; // [rsp+60h] [rbp-C0h]
  __int64 v75; // [rsp+70h] [rbp-B0h] BYREF
  int v76; // [rsp+78h] [rbp-A8h]
  __m128i v77; // [rsp+80h] [rbp-A0h] BYREF
  unsigned __int8 *v78; // [rsp+90h] [rbp-90h]
  __int64 v79; // [rsp+98h] [rbp-88h]
  __m128i v80; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v81; // [rsp+C0h] [rbp-60h] BYREF
  __m128i v82; // [rsp+D0h] [rbp-50h]
  unsigned __int8 *v83; // [rsp+E0h] [rbp-40h]
  unsigned __int64 v84; // [rsp+E8h] [rbp-38h]

  v8 = a1;
  v10 = *(_QWORD *)(a2 + 48);
  LOWORD(a1) = *(_WORD *)v10;
  v69 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(v10 + 8);
  LODWORD(v10) = *(_DWORD *)(a2 + 24);
  v11 = *(_QWORD *)(a2 + 80);
  v71 = v10;
  v75 = v11;
  if ( v11 )
    sub_B96E90((__int64)&v75, v11, 1);
  v76 = *(_DWORD *)(a2 + 72);
  v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL);
  v13 = *(_DWORD *)(v12 + 24);
  if ( v13 == 35 || v13 == 11 )
  {
    sub_3817F00((__int64)v8, a2, (_QWORD **)(*(_QWORD *)(v12 + 96) + 24LL), a3, (__int64)a4, a5);
    goto LABEL_23;
  }
  v64 = sub_3819010(v8, a2, a3, a4, a5);
  if ( v64 )
    goto LABEL_23;
  v63 = 210;
  if ( v71 != 190 )
    v63 = (v71 == 192) + 211;
  v14 = v8[1];
  v15 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*v8 + 592LL);
  if ( v15 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v81, *v8, *(_QWORD *)(v14 + 64), (unsigned __int16)a1, (__int64)v69);
    v16 = v81.m128i_u16[4];
    v17 = v82.m128i_i64[0];
  }
  else
  {
    v60 = (unsigned __int16)a1;
    v65 = v15(*v8, *(_QWORD *)(v14 + 64), (unsigned __int16)a1, (__int64)v69);
    v16 = v65;
    v17 = v33;
  }
  v18 = *v8;
  if ( v16 )
  {
    v19 = *(_BYTE *)(v63 + v18 + 500LL * v16 + 6414);
    if ( v19 )
      v64 = v19 == 4;
    else
      v64 = *(_QWORD *)(v18 + 8LL * v16 + 112) != 0;
  }
  v20 = v65;
  v66 = 1;
  LOWORD(v20) = v16;
  v62 = a3;
  v21 = v8;
  v22 = v17;
  v23 = v20;
  v61 = (__int64)a4;
  v24 = v16;
  while ( 1 )
  {
    v25 = v21[1];
    LOWORD(v23) = v24;
    v26 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v18 + 592LL);
    if ( v26 == sub_2D56A50 )
    {
      sub_2FE6CC0((__int64)&v81, v18, *(_QWORD *)(v25 + 64), v23, v22);
      LOWORD(v27) = v81.m128i_i16[4];
      v28 = v82.m128i_i64[0];
    }
    else
    {
      v27 = v26(v18, *(_QWORD *)(v25 + 64), v23, v22);
      v5 = v27;
    }
    if ( v24 == (_WORD)v27 && ((_WORD)v27 || v22 == v28) )
      break;
    LOWORD(v5) = v27;
    v18 = *v21;
    v22 = v28;
    v24 = v27;
    v23 = v5;
    ++v66;
  }
  v29 = *(_BOOL8 (__fastcall **)(__int64, __int64, __int64, int))(*(_QWORD *)*v21 + 584LL);
  if ( v29 == sub_2FE3150 )
  {
    if ( v66 != 1 )
    {
LABEL_21:
      sub_3824710(v21, a2, v62, v61);
      goto LABEL_23;
    }
  }
  else
  {
    v30 = v29(*v21, v21[1], a2, v66);
    if ( v30 == 1 )
      goto LABEL_21;
    v64 &= v30 != 2;
  }
  if ( !v64 )
  {
    if ( v71 == 190 )
    {
      switch ( (_WORD)a1 )
      {
        case 6:
          v31 = 0;
          v32 = 0;
          break;
        case 7:
          v31 = 1;
          v32 = 1;
          break;
        case 8:
          v31 = 2;
          v32 = 2;
          break;
        case 9:
          v31 = 3;
          v32 = 3;
          break;
        default:
          goto LABEL_50;
      }
    }
    else if ( v71 == 192 )
    {
      switch ( (_WORD)a1 )
      {
        case 6:
          v31 = 4;
          v32 = 4;
          break;
        case 7:
          v31 = 5;
          v32 = 5;
          break;
        case 8:
          v31 = 6;
          v32 = 6;
          break;
        case 9:
          v31 = 7;
          v32 = 7;
          break;
        default:
          goto LABEL_50;
      }
    }
    else
    {
      switch ( (_WORD)a1 )
      {
        case 6:
          v64 = 1;
          v31 = 8;
          v32 = 8;
          break;
        case 7:
          v64 = 1;
          v31 = 9;
          v32 = 9;
          break;
        case 8:
          v64 = 1;
          v31 = 10;
          v32 = 10;
          break;
        case 9:
          v64 = 1;
          v31 = 11;
          v32 = 11;
          break;
        default:
LABEL_50:
          if ( !(unsigned __int8)sub_3819B30(v21, a2, v62, v61) )
            BUG();
          goto LABEL_23;
      }
    }
    if ( *(_QWORD *)(*v21 + 8 * v31 + 525288) )
    {
      v73 = v32;
      v52 = sub_327FC40(*(_QWORD **)(v21[1] + 64), *(_DWORD *)(**(_QWORD **)(v21[1] + 24) + 172LL));
      v54 = sub_33FB310(
              v21[1],
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
              (__int64)&v75,
              v52,
              v53,
              a5);
      v55 = (_WORD *)*v21;
      WORD1(a1) = HIWORD(v60);
      v56 = _mm_loadu_si128((const __m128i *)*(_QWORD *)(a2 + 40));
      v78 = v54;
      v79 = v57;
      v58 = v21[1];
      v82 = 0u;
      LOBYTE(v83) = v64 & 1 | 4;
      v81 = 0u;
      v77 = v56;
      sub_3494590(
        (__int64)&v80,
        v55,
        v58,
        v73,
        (unsigned int)a1,
        v69,
        (__int64)&v77,
        2u,
        0,
        0,
        0,
        0,
        (char)v83,
        (__int64)&v75,
        0,
        0);
      sub_375BC20(v21, v80.m128i_i64[0], v80.m128i_i64[1], v62, v61, v56);
      goto LABEL_23;
    }
    goto LABEL_50;
  }
  v34 = *(_QWORD *)(a2 + 40);
  HIWORD(v37) = 0;
  v80.m128i_i64[0] = 0;
  v77.m128i_i64[0] = 0;
  v77.m128i_i32[2] = 0;
  v80.m128i_i32[2] = 0;
  sub_375E510((__int64)v21, *(_QWORD *)v34, *(_QWORD *)(v34 + 8), (__int64)&v77, (__int64)&v80);
  v35 = *(_QWORD *)(v77.m128i_i64[0] + 48) + 16LL * v77.m128i_u32[2];
  v36 = *(_QWORD *)(v35 + 8);
  LOWORD(v37) = *(_WORD *)v35;
  v67 = *v21;
  v38 = *(_QWORD *)(a2 + 40);
  v72 = v36;
  v39 = *(_QWORD *)(v38 + 48);
  v68 = *(unsigned __int8 **)(v38 + 40);
  v70 = *(_DWORD *)(v38 + 48);
  v40 = sub_2E79000(*(__int64 **)(v21[1] + 40));
  v41 = sub_2FE6750(v67, (unsigned __int16)v37, v72, v40);
  v43 = v42;
  v44 = v70;
  v45 = v41;
  v46 = *((_QWORD *)v68 + 6) + 16LL * v70;
  if ( *(_WORD *)v46 != (_WORD)v45 || *(_QWORD *)(v46 + 8) != v43 && !*(_WORD *)v46 )
  {
    v68 = sub_33FB310(v21[1], (__int64)v68, v39, (__int64)&v75, v45, v43, a5);
    v44 = (unsigned int)v44;
  }
  v47 = (__int64 *)v21[1];
  v48 = _mm_loadu_si128(&v80);
  v81 = _mm_loadu_si128(&v77);
  v82 = v48;
  v83 = v68;
  v84 = v44 | v39 & 0xFFFFFFFF00000000LL;
  v49 = (unsigned int *)sub_33E5110(v47, v37, v72, v37, v72);
  *((_QWORD *)&v59 + 1) = 3;
  *(_QWORD *)&v59 = &v81;
  v74 = sub_3411630(v47, v63, (__int64)&v75, v49, v50, (__int64)&v81, v59);
  *(_QWORD *)v62 = v74;
  *(_DWORD *)(v62 + 8) = v51;
  *(_QWORD *)v61 = v74;
  *(_DWORD *)(v61 + 8) = 1;
LABEL_23:
  if ( v75 )
    sub_B91220((__int64)&v75, v75);
}
