// Function: sub_36DE090
// Address: 0x36de090
//
__int64 __fastcall sub_36DE090(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  const __m128i *v6; // rdx
  __int64 v7; // rax
  _QWORD *v8; // r8
  unsigned __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rcx
  int v12; // r14d
  __int64 v13; // r13
  unsigned int *v14; // rdx
  unsigned int **v15; // rdi
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // r8
  __int64 v19; // r9
  __int32 v20; // edx
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  __m128i *v23; // rax
  __m128i v24; // xmm2
  unsigned int *v25; // rdi
  unsigned __int16 v27; // di
  unsigned __int64 v28; // rax
  unsigned __int64 v29; // rsi
  __int64 v30; // rax
  _QWORD *v31; // rdi
  __int64 v32; // r13
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  unsigned __int16 v36; // di
  unsigned __int64 v37; // rax
  unsigned __int64 v38; // rax
  __int16 v39; // ax
  unsigned __int16 v40; // di
  __int128 v41; // [rsp-10h] [rbp-130h]
  __int64 v42; // [rsp+8h] [rbp-118h]
  __m128i v43; // [rsp+10h] [rbp-110h] BYREF
  _QWORD *v44; // [rsp+20h] [rbp-100h]
  unsigned int **v45; // [rsp+28h] [rbp-F8h]
  __int32 v46; // [rsp+34h] [rbp-ECh]
  __int64 v47; // [rsp+38h] [rbp-E8h]
  unsigned __int64 v48; // [rsp+40h] [rbp-E0h]
  __int64 v49; // [rsp+48h] [rbp-D8h]
  __int64 v50; // [rsp+50h] [rbp-D0h] BYREF
  int v51; // [rsp+58h] [rbp-C8h]
  __m128i v52; // [rsp+60h] [rbp-C0h] BYREF
  __m128i v53; // [rsp+70h] [rbp-B0h] BYREF
  unsigned int *v54; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v55; // [rsp+88h] [rbp-98h]
  _BYTE v56[144]; // [rsp+90h] [rbp-90h] BYREF

  v5 = *(_QWORD *)(a2 + 80);
  v50 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v50, v5, 1);
  v6 = *(const __m128i **)(a2 + 40);
  v51 = *(_DWORD *)(a2 + 72);
  v7 = *(_QWORD *)(v6[2].m128i_i64[1] + 96);
  v8 = *(_QWORD **)(v7 + 24);
  if ( *(_DWORD *)(v7 + 32) > 0x40u )
    v8 = (_QWORD *)*v8;
  LODWORD(v9) = 0;
  v10 = (unsigned int)(*(_DWORD *)(a2 + 24) - 576);
  if ( (unsigned int)v10 <= 2 )
  {
    v11 = v6->m128i_i64[0];
    v54 = (unsigned int *)v56;
    v47 = v11;
    v46 = v6->m128i_i32[2];
    v12 = dword_4501008[v10];
    v55 = 0x600000000LL;
    if ( v12 )
    {
      a3 = _mm_loadu_si128(v6 + 5);
      v13 = 120;
      v14 = (unsigned int *)v56;
      v15 = &v54;
      v16 = 40LL * (unsigned int)(v12 - 1) + 120;
      v17 = 0;
      while ( 1 )
      {
        *(__m128i *)&v14[4 * v17] = a3;
        v17 = (unsigned int)(v55 + 1);
        LODWORD(v55) = v55 + 1;
        if ( v13 == v16 )
          break;
        a3 = _mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + v13));
        if ( v17 + 1 > (unsigned __int64)HIDWORD(v55) )
        {
          v42 = v16;
          v44 = v8;
          v45 = v15;
          v43 = a3;
          sub_C8D5F0((__int64)v15, v56, v17 + 1, 0x10u, (__int64)v8, v16);
          v17 = (unsigned int)v55;
          v16 = v42;
          v8 = v44;
          a3 = _mm_load_si128(&v43);
          v15 = v45;
        }
        v14 = v54;
        v13 += 40;
      }
    }
    v52.m128i_i64[0] = (__int64)sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v8, (__int64)&v50, 7, 0, 1u, a3, 0);
    v52.m128i_i32[2] = v20;
    v53.m128i_i64[0] = v47;
    v53.m128i_i32[2] = v46;
    v21 = (unsigned int)v55;
    v22 = (unsigned int)v55 + 2LL;
    if ( v22 > HIDWORD(v55) )
    {
      sub_C8D5F0((__int64)&v54, v56, v22, 0x10u, v18, v19);
      v21 = (unsigned int)v55;
    }
    v23 = (__m128i *)&v54[4 * v21];
    *v23 = _mm_load_si128(&v52);
    v24 = _mm_load_si128(&v53);
    LODWORD(v55) = v55 + 2;
    v23[1] = v24;
    switch ( v12 )
    {
      case 2:
        v40 = *(_WORD *)(a2 + 96);
        v52.m128i_i64[0] = 0x100001292LL;
        v49 = 0x100001295LL;
        v28 = sub_36D6650(v40, 4758, 4755, 4756, 0x100001295LL, 4753, 0x100001292LL);
        break;
      case 4:
        v52.m128i_i8[4] = 0;
        v27 = *(_WORD *)(a2 + 96);
        BYTE4(v49) = 0;
        v28 = sub_36D6650(v27, 4762, 4760, 4761, v49, 4759, v52.m128i_i64[0]);
        break;
      case 1:
        v36 = *(_WORD *)(a2 + 96);
        v52.m128i_i64[0] = 0x10000128ALL;
        v49 = 0x10000128DLL;
        v37 = sub_36D6650(v36, 4750, 4747, 4748, 0x10000128DLL, 4745, 0x10000128ALL);
        v25 = v54;
        v29 = v37;
        v38 = HIDWORD(v37);
        v48 = v29;
        LODWORD(v9) = v38;
        if ( (_DWORD)v29 == 4750 && (_BYTE)v38 )
        {
          LODWORD(v29) = 4751;
          v39 = *(_WORD *)(*(_QWORD *)(*(_QWORD *)v54 + 48LL) + 16LL * v54[2]);
          if ( v39 != 7 )
            LODWORD(v29) = 2 * (v39 == 8) + 4750;
LABEL_26:
          LODWORD(v9) = 1;
          *((_QWORD *)&v41 + 1) = (unsigned int)v55;
          *(_QWORD *)&v41 = v25;
          v30 = sub_33F7800(*(_QWORD **)(a1 + 64), v29, (__int64)&v50, 1u, 0, *(_QWORD *)(a1 + 64), v41);
          v31 = *(_QWORD **)(a1 + 64);
          v32 = v30;
          v52.m128i_i64[0] = *(_QWORD *)(a2 + 112);
          sub_33E4DA0(v31, v30, v52.m128i_i64, 1);
          sub_34158F0(*(_QWORD *)(a1 + 64), a2, v32, v33, v34, v35);
          sub_3421DB0(v32);
          sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
          v25 = v54;
LABEL_18:
          if ( v25 != (unsigned int *)v56 )
            _libc_free((unsigned __int64)v25);
          goto LABEL_20;
        }
LABEL_25:
        if ( !(_BYTE)v9 )
          goto LABEL_18;
        goto LABEL_26;
      default:
        v25 = v54;
        LODWORD(v9) = 0;
        goto LABEL_18;
    }
    v25 = v54;
    LODWORD(v29) = v28;
    v9 = HIDWORD(v28);
    goto LABEL_25;
  }
LABEL_20:
  if ( v50 )
    sub_B91220((__int64)&v50, v50);
  return (unsigned int)v9;
}
