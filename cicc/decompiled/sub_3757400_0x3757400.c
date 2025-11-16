// Function: sub_3757400
// Address: 0x3757400
//
__int64 __fastcall sub_3757400(__int64 *a1, __int64 a2, __m128i *a3, unsigned __int8 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rdx
  int v11; // edx
  __int64 *v12; // rax
  __m128i v13; // xmm0
  __m128i v14; // xmm1
  __int64 v15; // rax
  __int64 v16; // rdi
  char v17; // dl
  __int64 v18; // rsi
  __int64 (__fastcall *v19)(__int64, unsigned __int16); // rax
  __int64 v20; // rsi
  __int64 v21; // rdi
  __int64 (__fastcall *v22)(__int64, __int64); // rax
  __int64 v23; // rdi
  int v24; // eax
  unsigned __int8 *v25; // rsi
  __int64 v26; // r13
  __int64 v27; // r13
  unsigned __int8 *v28; // rbx
  _QWORD *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  _QWORD *v32; // rax
  unsigned __int8 v33; // r10
  __int64 v34; // rbx
  __int64 *v35; // r12
  __int64 v36; // rdx
  __int64 v37; // rax
  unsigned __int64 *v39; // rax
  __int64 v40; // rdx
  _QWORD *v41; // rbx
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 (__fastcall *v44)(__int64, unsigned __int16); // rcx
  __int64 v45; // r13
  unsigned __int64 v46; // rsi
  int v47; // r12d
  unsigned __int8 *v48; // rsi
  __int64 v49; // r13
  __int64 v50; // rcx
  __int64 *v51; // rsi
  __int64 v52; // rdi
  __int64 v53; // rdx
  __int32 v54; // eax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rsi
  __int64 v57; // rdi
  __int64 (*v58)(); // rax
  __int64 v59; // rdi
  __int32 v60; // eax
  unsigned __int8 *v61; // rsi
  __int64 v62; // r12
  __int64 v63; // r12
  __int64 *v64; // rsi
  __int64 v65; // rdi
  _QWORD *v66; // rax
  __int64 v67; // rdx
  __int64 v68; // [rsp+10h] [rbp-F0h]
  _QWORD *v69; // [rsp+18h] [rbp-E8h]
  _QWORD *v70; // [rsp+18h] [rbp-E8h]
  _QWORD *v71; // [rsp+18h] [rbp-E8h]
  __int64 v72; // [rsp+30h] [rbp-D0h]
  __int64 v74; // [rsp+38h] [rbp-C8h]
  unsigned int v76; // [rsp+50h] [rbp-B0h]
  _QWORD *v77; // [rsp+50h] [rbp-B0h]
  unsigned __int8 v78; // [rsp+58h] [rbp-A8h]
  __int32 v79; // [rsp+5Ch] [rbp-A4h]
  int v80; // [rsp+60h] [rbp-A0h] BYREF
  int v81; // [rsp+64h] [rbp-9Ch] BYREF
  unsigned __int8 *v82; // [rsp+68h] [rbp-98h] BYREF
  unsigned __int8 *v83; // [rsp+70h] [rbp-90h] BYREF
  __int64 v84; // [rsp+78h] [rbp-88h]
  unsigned __int8 *v85; // [rsp+80h] [rbp-80h] BYREF
  __int64 v86; // [rsp+88h] [rbp-78h]
  __int64 v87; // [rsp+90h] [rbp-70h] BYREF
  __m128i v88; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v89; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v90; // [rsp+B8h] [rbp-48h]
  __int64 v91; // [rsp+C0h] [rbp-40h]

  v78 = a5;
  v76 = ~*(_DWORD *)(a2 + 24);
  v8 = *(_QWORD *)(a2 + 56);
  if ( v8 )
  {
    while ( 1 )
    {
      v9 = *(_QWORD *)(v8 + 16);
      if ( *(_DWORD *)(v9 + 24) == 49 )
      {
        v10 = *(_QWORD *)(v9 + 40);
        if ( a2 == *(_QWORD *)(v10 + 80) )
        {
          v11 = *(_DWORD *)(*(_QWORD *)(v10 + 40) + 96LL);
          if ( v11 < 0 )
            break;
        }
      }
      v8 = *(_QWORD *)(v8 + 32);
      if ( !v8 )
        goto LABEL_8;
    }
    v79 = v11;
  }
  else
  {
LABEL_8:
    v79 = 0;
  }
  if ( v76 != 8 )
  {
    if ( v76 != 9 && v76 != 12 )
      BUG();
    v12 = *(__int64 **)(a2 + 40);
    v13 = _mm_loadu_si128((const __m128i *)v12);
    v14 = _mm_loadu_si128((const __m128i *)(v12 + 5));
    v68 = *v12;
    v15 = *(_QWORD *)(v12[10] + 96);
    if ( *(_DWORD *)(v15 + 32) <= 0x40u )
      v72 = *(_QWORD *)(v15 + 24);
    else
      v72 = **(_QWORD **)(v15 + 24);
    v16 = a1[4];
    v17 = *(_BYTE *)(a2 + 32);
    v18 = **(unsigned __int16 **)(a2 + 48);
    v19 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v16 + 552LL);
    if ( v19 == sub_2EC09E0 )
      v20 = *(_QWORD *)(v16 + 8 * v18 + 112);
    else
      v20 = ((__int64 (__fastcall *)(__int64, __int64, bool))v19)(v16, v18, (v17 & 4) != 0);
    v21 = a1[3];
    v22 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v21 + 272LL);
    if ( v22 != sub_2E85430 )
      v20 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v22)(v21, v20, (unsigned int)v72);
    v23 = a1[1];
    if ( !v79
      || (v24 = *(_DWORD *)(*(_QWORD *)(v20 + 8)
                          + 4
                          * ((unsigned __int64)*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 56)
                                                                                           + 16LL * (v79 & 0x7FFFFFFF))
                                                                               & 0xFFFFFFFFFFFFFFF8LL)
                                                                   + 24LL) >> 5)),
          !_bittest(
             &v24,
             *(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v23 + 56) + 16LL * (v79 & 0x7FFFFFFF))
                                             & 0xFFFFFFFFFFFFFFF8LL)
                                 + 24LL))) )
    {
      v79 = sub_2EC06C0(v23, v20, byte_3F871B3, 0, a5, a6);
    }
    v25 = *(unsigned __int8 **)(a2 + 80);
    v26 = *(_QWORD *)(a1[2] + 8);
    v82 = v25;
    v27 = v26 - 40LL * v76;
    if ( v25 )
    {
      sub_B96E90((__int64)&v82, (__int64)v25, 1);
      v85 = v82;
      if ( v82 )
      {
        sub_B976B0((__int64)&v82, v82, (__int64)&v85);
        v86 = 0;
        v82 = 0;
        v28 = (unsigned __int8 *)*a1;
        v87 = 0;
        v83 = v85;
        if ( v85 )
          sub_B96E90((__int64)&v83, (__int64)v85, 1);
        goto LABEL_24;
      }
    }
    else
    {
      v85 = 0;
    }
    v28 = (unsigned __int8 *)*a1;
    v86 = 0;
    v87 = 0;
    v83 = 0;
LABEL_24:
    v29 = sub_2E7B380(v28, v27, &v83, 0);
    if ( v86 )
    {
      v69 = v29;
      sub_2E882B0((__int64)v29, (__int64)v28, v86);
      v29 = v69;
    }
    if ( v87 )
    {
      v70 = v29;
      sub_2E88680((__int64)v29, (__int64)v28, v87);
      v29 = v70;
    }
    v88.m128i_i64[0] = 0x10000000;
    v71 = v29;
    v89 = 0;
    v88.m128i_i32[2] = v79;
    v90 = 0;
    v91 = 0;
    sub_2E8EAD0((__int64)v29, (__int64)v28, &v88);
    v30 = (__int64)v71;
    if ( v83 )
    {
      sub_B91220((__int64)&v83, (__int64)v83);
      v30 = (__int64)v71;
    }
    v83 = v28;
    v84 = v30;
    if ( v85 )
      sub_B91220((__int64)&v85, (__int64)v85);
    if ( v82 )
      sub_B91220((__int64)&v82, (__int64)v82);
    if ( v76 == 12 )
    {
      v31 = *(_QWORD *)(v68 + 96);
      v32 = *(_QWORD **)(v31 + 24);
      if ( *(_DWORD *)(v31 + 32) > 0x40u )
        v32 = (_QWORD *)*v32;
      v88.m128i_i64[0] = 1;
      v89 = 0;
      v90 = (unsigned __int64)v32;
      sub_2E8EAD0(v84, (__int64)v83, &v88);
      v33 = v78;
    }
    else
    {
      sub_3752760(a1, (__int64 *)&v83, v13.m128i_u64[0], v13.m128i_u32[2], 0, 0, (__int64)a3, 0, a4, v78);
      v33 = v78;
    }
    sub_3752760(a1, (__int64 *)&v83, v14.m128i_u64[0], v14.m128i_u32[2], 0, 0, (__int64)a3, 0, a4, v33);
    v88.m128i_i64[0] = 1;
    v90 = (unsigned int)v72;
    v89 = 0;
    sub_2E8EAD0(v84, (__int64)v83, &v88);
    v34 = v84;
    v35 = (__int64 *)a1[6];
    sub_2E31040((__int64 *)(a1[5] + 40), v84);
    v36 = *v35;
    v37 = *(_QWORD *)v34;
    *(_QWORD *)(v34 + 8) = v35;
    v36 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v34 = v36 | v37 & 7;
    *(_QWORD *)(v36 + 8) = v34;
    *v35 = *v35 & 7 | v34;
    goto LABEL_39;
  }
  v39 = *(unsigned __int64 **)(a2 + 40);
  v40 = *(_QWORD *)(v39[5] + 96);
  v41 = *(_QWORD **)(v40 + 24);
  if ( *(_DWORD *)(v40 + 32) > 0x40u )
    v41 = (_QWORD *)*v41;
  v42 = a1[4];
  v43 = **(unsigned __int16 **)(a2 + 48);
  v44 = *(__int64 (__fastcall **)(__int64, unsigned __int16))(*(_QWORD *)v42 + 552LL);
  if ( v44 == sub_2EC09E0 )
  {
    v45 = *(_QWORD *)(v42 + 8 * v43 + 112);
  }
  else
  {
    v45 = ((__int64 (__fastcall *)(__int64, __int64, bool))v44)(v42, v43, (*(_BYTE *)(a2 + 32) & 4) != 0);
    v39 = *(unsigned __int64 **)(a2 + 40);
  }
  v46 = *v39;
  if ( *(_DWORD *)(*v39 + 24) == 9 )
  {
    v47 = *(_DWORD *)(v46 + 96);
    if ( (unsigned int)(v47 - 1) <= 0x3FFFFFFE )
    {
      v80 = 0;
      v81 = 0;
LABEL_50:
      if ( !v79 )
        v79 = sub_2EC06C0(a1[1], v45, byte_3F871B3, 0, a5, a6);
      v48 = *(unsigned __int8 **)(a2 + 80);
      v49 = *(_QWORD *)(a1[2] + 8);
      v85 = v48;
      v50 = v49 - 800;
      if ( v48 )
      {
        sub_B96E90((__int64)&v85, (__int64)v48, 1);
        v50 = v49 - 800;
        v88.m128i_i64[0] = (__int64)v85;
        if ( v85 )
        {
          sub_B976B0((__int64)&v85, v85, (__int64)&v88);
          v85 = 0;
          v50 = v49 - 800;
        }
      }
      else
      {
        v88.m128i_i64[0] = 0;
      }
      v51 = (__int64 *)a1[6];
      v52 = a1[5];
      v88.m128i_i64[1] = 0;
      v89 = 0;
      v77 = sub_2F26260(v52, v51, v88.m128i_i64, v50, v79);
      v74 = v53;
      if ( v88.m128i_i64[0] )
        sub_B91220((__int64)&v88, v88.m128i_i64[0]);
      if ( v85 )
        sub_B91220((__int64)&v85, (__int64)v85);
      if ( v47 < 0 )
      {
        v89 = 0;
        v88.m128i_i32[2] = v47;
        v90 = 0;
        v91 = 0;
        v88.m128i_i64[0] = (unsigned __int16)((unsigned __int16)v41 & 0xFFF) << 8;
      }
      else
      {
        v54 = sub_E91CF0((_QWORD *)a1[3], v47, (int)v41);
        v88.m128i_i64[0] = 0;
        v89 = 0;
        v88.m128i_i32[2] = v54;
        v90 = 0;
        v91 = 0;
      }
      sub_2E8EAD0(v74, (__int64)v77, &v88);
      goto LABEL_39;
    }
  }
  else
  {
    v47 = sub_3752000(a1, v46, v39[1], (__int64)a3, a5, a6);
  }
  v55 = sub_2EBEE10(a1[1], v47);
  v80 = 0;
  v81 = 0;
  v56 = v55;
  if ( !v55
    || (v57 = a1[2], v58 = *(__int64 (**)())(*(_QWORD *)v57 + 80LL), v58 == sub_2F28CC0)
    || !((unsigned __int8 (__fastcall *)(__int64, unsigned __int64, int *, int *, unsigned __int8 **))v58)(
          v57,
          v56,
          &v80,
          &v81,
          &v82)
    || (_DWORD)v82 != (_DWORD)v41
    || (v59 = a1[1], v45 != (*(_QWORD *)(*(_QWORD *)(v59 + 56) + 16LL * (v80 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL)) )
  {
    if ( v47 < 0 )
      v47 = sub_3752FF0(
              a1,
              v47,
              (unsigned int)v41,
              *(_WORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 48LL)
                       + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 8LL)),
              (*(_BYTE *)(a2 + 32) & 4) != 0,
              (unsigned __int8 **)(a2 + 80));
    goto LABEL_50;
  }
  v60 = sub_2EC06C0(v59, v45, byte_3F871B3, 0, a5, a6);
  v61 = *(unsigned __int8 **)(a2 + 80);
  v79 = v60;
  v62 = *(_QWORD *)(a1[2] + 8);
  v83 = v61;
  v63 = v62 - 800;
  if ( v61 )
  {
    sub_B96E90((__int64)&v83, (__int64)v61, 1);
    v85 = v83;
    if ( v83 )
    {
      sub_B976B0((__int64)&v83, v83, (__int64)&v85);
      v83 = 0;
    }
  }
  else
  {
    v85 = 0;
  }
  v64 = (__int64 *)a1[6];
  v65 = a1[5];
  v86 = 0;
  v87 = 0;
  v66 = sub_2F26260(v65, v64, (__int64 *)&v85, v63, v79);
  v88.m128i_i64[0] = 0;
  v89 = 0;
  v88.m128i_i32[2] = v80;
  v90 = 0;
  v91 = 0;
  sub_2E8EAD0(v67, (__int64)v66, &v88);
  if ( v85 )
    sub_B91220((__int64)&v85, (__int64)v85);
  if ( v83 )
    sub_B91220((__int64)&v83, (__int64)v83);
  sub_2EBF120(a1[1], v80);
LABEL_39:
  v85 = (unsigned __int8 *)a2;
  LODWORD(v86) = 0;
  LODWORD(v87) = v79;
  return sub_3755010((__int64)&v88, a3, (unsigned __int64 *)&v85, &v87);
}
