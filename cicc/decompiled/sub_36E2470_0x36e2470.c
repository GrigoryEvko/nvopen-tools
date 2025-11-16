// Function: sub_36E2470
// Address: 0x36e2470
//
__int64 __fastcall sub_36E2470(__int64 a1, __int64 a2, __m128i a3)
{
  __int64 result; // rax
  unsigned __int16 v4; // r15
  unsigned int v7; // ebx
  __int64 v8; // rsi
  int v9; // eax
  const __m128i *v10; // rdx
  __m128i v11; // xmm3
  __int64 v12; // rax
  char v13; // cl
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // rdx
  __int64 v17; // rax
  bool v18; // cc
  _QWORD *v19; // rax
  int v20; // r11d
  int v21; // eax
  const __m128i *v22; // rdx
  __int64 v23; // rax
  unsigned __int16 v24; // r15
  __int64 v25; // r11
  __int64 v26; // r10
  __int64 v27; // rax
  unsigned __int64 v28; // rdx
  unsigned __int8 *v29; // rax
  __int64 v30; // rdi
  int v31; // edx
  unsigned __int8 *v32; // rax
  __int64 v33; // rdi
  int v34; // edx
  unsigned __int8 *v35; // rax
  __int64 v36; // rdi
  int v37; // edx
  unsigned __int8 *v38; // rax
  __int64 v39; // rdi
  int v40; // edx
  unsigned __int8 *v41; // rax
  __int64 v42; // rdi
  int v43; // edx
  unsigned __int8 *v44; // rax
  __m128i v45; // xmm0
  __m128i v46; // xmm1
  __m128i v47; // xmm2
  int v48; // eax
  __int64 v49; // r9
  int v50; // edx
  __int64 v51; // rax
  int v52; // esi
  __int64 v53; // rax
  _QWORD *v54; // rdi
  __int64 v55; // r14
  __int64 v56; // rcx
  __int64 v57; // r8
  __int64 v58; // r9
  __int64 v59; // rax
  unsigned int v60; // r15d
  __int64 v61; // [rsp+8h] [rbp-158h]
  unsigned int v62; // [rsp+18h] [rbp-148h]
  __int64 v63; // [rsp+18h] [rbp-148h]
  unsigned int v64; // [rsp+20h] [rbp-140h]
  __int64 v65; // [rsp+20h] [rbp-140h]
  __int64 v66; // [rsp+28h] [rbp-138h]
  unsigned __int16 v67; // [rsp+36h] [rbp-12Ah]
  unsigned __int64 v68; // [rsp+38h] [rbp-128h]
  unsigned __int8 v69; // [rsp+38h] [rbp-128h]
  __int64 v70; // [rsp+58h] [rbp-108h] BYREF
  __int64 v71; // [rsp+60h] [rbp-100h] BYREF
  int v72; // [rsp+68h] [rbp-F8h]
  __m128i v73; // [rsp+70h] [rbp-F0h] BYREF
  __m128i v74; // [rsp+80h] [rbp-E0h] BYREF
  __m128i v75; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v76; // [rsp+A0h] [rbp-C0h] BYREF
  int v77; // [rsp+A8h] [rbp-B8h]
  unsigned __int8 *v78; // [rsp+B0h] [rbp-B0h]
  int v79; // [rsp+B8h] [rbp-A8h]
  unsigned __int8 *v80; // [rsp+C0h] [rbp-A0h]
  int v81; // [rsp+C8h] [rbp-98h]
  unsigned __int8 *v82; // [rsp+D0h] [rbp-90h]
  int v83; // [rsp+D8h] [rbp-88h]
  unsigned __int8 *v84; // [rsp+E0h] [rbp-80h]
  int v85; // [rsp+E8h] [rbp-78h]
  unsigned __int8 *v86; // [rsp+F0h] [rbp-70h]
  int v87; // [rsp+F8h] [rbp-68h]
  __m128i v88; // [rsp+100h] [rbp-60h]
  __m128i v89; // [rsp+110h] [rbp-50h]
  __m128i v90; // [rsp+120h] [rbp-40h]

  result = 0;
  v4 = *(_WORD *)(a2 + 96);
  if ( v4 )
  {
    v7 = sub_36D7800(*(_QWORD *)(a2 + 112));
    if ( *(_DWORD *)(*(_QWORD *)(a1 + 1136) + 344LL) > 0x1Fu
      && v7 == 1
      && ((*(_BYTE *)(a2 + 32) & 0x40) != 0 || (unsigned __int8)sub_36D7A70(a2, **(_QWORD **)(a1 + 40))) )
    {
      return sub_36E0E00(a1, a2, a3);
    }
    v8 = *(_QWORD *)(a2 + 80);
    v71 = v8;
    if ( v8 )
    {
      sub_B96E90((__int64)&v71, v8, 1);
      v9 = *(_DWORD *)(a2 + 72);
      v10 = *(const __m128i **)(a2 + 40);
      v72 = v9;
      v11 = _mm_loadu_si128(v10);
      v76 = v71;
      v73 = v11;
      if ( v71 )
      {
        sub_B96E90((__int64)&v76, v71, 1);
        v9 = v72;
      }
    }
    else
    {
      v9 = *(_DWORD *)(a2 + 72);
      v22 = *(const __m128i **)(a2 + 40);
      v72 = v9;
      v76 = 0;
      v73 = _mm_loadu_si128(v22);
    }
    v77 = v9;
    v68 = sub_36E1BC0(a1, (__int64)&v76, (__int64)&v73, a2);
    if ( v76 )
      sub_B91220((__int64)&v76, v76);
    if ( (unsigned __int16)(v4 - 17) <= 0xD3u )
      v4 = word_4456580[v4 - 1];
    if ( v4 <= 1u || (unsigned __int16)(v4 - 504) <= 7u )
      BUG();
    v12 = 16LL * (v4 - 1);
    v13 = byte_444C4A0[v12 + 8];
    v14 = *(_QWORD *)&byte_444C4A0[v12];
    LOBYTE(v77) = v13;
    v76 = v14;
    v15 = sub_CA1930(&v76);
    v16 = *(_QWORD *)(a2 + 40);
    v64 = v15;
    v17 = *(_QWORD *)(*(_QWORD *)(v16 + 40LL * (unsigned int)(*(_DWORD *)(a2 + 64) - 1)) + 96LL);
    v18 = *(_DWORD *)(v17 + 32) <= 0x40u;
    v19 = *(_QWORD **)(v17 + 24);
    if ( !v18 )
      v19 = (_QWORD *)*v19;
    v20 = 1;
    if ( (_DWORD)v19 != 2 )
      v20 = sub_36D79E0(v4, 0);
    v21 = *(_DWORD *)(a2 + 24);
    switch ( v21 )
    {
      case 549:
        v66 = 4;
        break;
      case 550:
        v59 = *(_QWORD *)(a1 + 1136);
        if ( *(_DWORD *)(v59 + 344) <= 0x63u || *(_DWORD *)(v59 + 336) <= 0x57u )
          goto LABEL_21;
        v66 = 8;
        break;
      case 548:
        v66 = 2;
        break;
      default:
LABEL_21:
        result = 0;
        goto LABEL_22;
    }
    v23 = *(_QWORD *)(a2 + 48);
    v62 = v20;
    v67 = *(_WORD *)v23;
    v24 = *(_WORD *)v23;
    if ( (unsigned __int8)sub_307AB50(*(_WORD *)v23, *(_QWORD *)(v23 + 8), v16) || (v25 = v62, v24 == 37) )
    {
      v25 = 3;
      v26 = 32;
      v67 = 7;
    }
    else
    {
      v60 = v64;
      if ( v64 < 8 )
        v60 = 8;
      v26 = v60;
    }
    v27 = *(_QWORD *)(a2 + 40);
    v74.m128i_i32[2] = 0;
    v63 = v26;
    v75.m128i_i32[2] = 0;
    v28 = *(_QWORD *)(v27 + 48);
    v74.m128i_i64[0] = 0;
    v75.m128i_i64[0] = 0;
    v65 = v25;
    sub_36DF750(a1, *(_QWORD *)(v27 + 40), v28, (__int64)&v75, (__int64)&v74, a3);
    v29 = sub_3400BD0(*(_QWORD *)(a1 + 64), (unsigned int)v68, (__int64)&v71, 7, 0, 1u, a3, 0);
    v30 = *(_QWORD *)(a1 + 64);
    v77 = v31;
    v76 = (__int64)v29;
    v32 = sub_3400BD0(v30, HIDWORD(v68), (__int64)&v71, 7, 0, 1u, a3, 0);
    v33 = *(_QWORD *)(a1 + 64);
    v79 = v34;
    v78 = v32;
    v35 = sub_3400BD0(v33, v7, (__int64)&v71, 7, 0, 1u, a3, 0);
    v36 = *(_QWORD *)(a1 + 64);
    v81 = v37;
    v80 = v35;
    v38 = sub_3400BD0(v36, v66, (__int64)&v71, 7, 0, 1u, a3, 0);
    v39 = *(_QWORD *)(a1 + 64);
    v83 = v40;
    v82 = v38;
    v41 = sub_3400BD0(v39, v65, (__int64)&v71, 7, 0, 1u, a3, 0);
    v42 = *(_QWORD *)(a1 + 64);
    v85 = v43;
    v84 = v41;
    v44 = sub_3400BD0(v42, v63, (__int64)&v71, 7, 0, 1u, a3, 0);
    v45 = _mm_loadu_si128(&v75);
    v46 = _mm_loadu_si128(&v74);
    v47 = _mm_loadu_si128(&v73);
    v86 = v44;
    v48 = *(_DWORD *)(a2 + 24);
    v49 = v61;
    v87 = v50;
    v88 = v45;
    v89 = v46;
    v90 = v47;
    switch ( v48 )
    {
      case 549:
        v70 = 0x100000A67LL;
        v51 = sub_36D6650(v67, 2672, 2665, 2667, 0x100000A6ELL, 2660, 0x100000A67LL);
        break;
      case 550:
        if ( v67 == 7 )
        {
          v52 = 2668;
        }
        else
        {
          if ( v67 != 12 )
            goto LABEL_21;
          v52 = 2661;
        }
        goto LABEL_37;
      case 548:
        v70 = 0x100000A66LL;
        v51 = sub_36D6650(v67, 2671, 2664, 2666, 0x100000A6DLL, 2659, 0x100000A66LL);
        break;
      default:
        goto LABEL_21;
    }
    v52 = v51;
    if ( !BYTE4(v51) )
      goto LABEL_21;
LABEL_37:
    v53 = sub_33E66D0(
            *(_QWORD **)(a1 + 64),
            v52,
            (__int64)&v71,
            *(_QWORD *)(a2 + 48),
            *(unsigned int *)(a2 + 68),
            v49,
            (unsigned __int64 *)&v76,
            9);
    v54 = *(_QWORD **)(a1 + 64);
    v55 = v53;
    v70 = *(_QWORD *)(a2 + 112);
    sub_33E4DA0(v54, v53, &v70, 1);
    sub_34158F0(*(_QWORD *)(a1 + 64), a2, v55, v56, v57, v58);
    sub_3421DB0(v55);
    sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
    result = 1;
LABEL_22:
    if ( v71 )
    {
      v69 = result;
      sub_B91220((__int64)&v71, v71);
      return v69;
    }
  }
  return result;
}
