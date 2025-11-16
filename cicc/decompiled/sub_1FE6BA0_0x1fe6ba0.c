// Function: sub_1FE6BA0
// Address: 0x1fe6ba0
//
__int64 __fastcall sub_1FE6BA0(
        __int64 *a1,
        __int64 *a2,
        unsigned __int64 a3,
        unsigned int a4,
        unsigned int a5,
        __int64 a6,
        __int64 a7,
        char a8,
        unsigned __int8 a9,
        unsigned int a10)
{
  __int16 v14; // ax
  __int64 v15; // rdx
  bool v16; // r15
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rax
  unsigned int v20; // edx
  __int64 *v21; // rax
  __int64 v22; // rsi
  __int32 v24; // r10d
  unsigned __int8 *v25; // rcx
  __int64 v26; // rsi
  __int64 v27; // rdi
  __int64 v28; // r11
  __int64 (__fastcall *v29)(__int64, unsigned __int8); // rax
  _QWORD *v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // r9
  __int64 v33; // rdi
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdi
  int v38; // eax
  __int64 v39; // rsi
  _QWORD *v40; // r12
  _QWORD *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rsi
  __int64 v46; // r14
  __int64 v47; // r12
  __int64 v48; // rdx
  __int64 v49; // rax
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rax
  __int64 v53; // rax
  int v54; // eax
  __int64 v55; // rdi
  __int64 v56; // rsi
  int v57; // eax
  int v58; // ebx
  __int64 v59; // rax
  unsigned int v60; // edx
  __int64 v61; // r15
  __int64 v62; // rsi
  __int64 v63; // rdi
  int v64; // eax
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  int v69; // edx
  __int64 v70; // [rsp+8h] [rbp-88h]
  int v71; // [rsp+14h] [rbp-7Ch]
  _QWORD *v72; // [rsp+18h] [rbp-78h]
  __int64 *v73; // [rsp+18h] [rbp-78h]
  _QWORD *v74; // [rsp+20h] [rbp-70h]
  __int32 v75; // [rsp+20h] [rbp-70h]
  __int64 v76; // [rsp+20h] [rbp-70h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  __int64 v78; // [rsp+28h] [rbp-68h]
  __int64 v79; // [rsp+28h] [rbp-68h]
  __int32 v80; // [rsp+28h] [rbp-68h]
  __int32 v81; // [rsp+28h] [rbp-68h]
  unsigned int v82; // [rsp+28h] [rbp-68h]
  __m128i v83; // [rsp+30h] [rbp-60h] BYREF
  __int64 v84; // [rsp+40h] [rbp-50h]
  __int64 v85; // [rsp+48h] [rbp-48h]
  __int64 v86; // [rsp+50h] [rbp-40h]

  v14 = *(_WORD *)(a3 + 24);
  if ( v14 < 0 )
    return sub_1FE67D0((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  v15 = (unsigned int)v14;
  if ( v14 == 10 || (_DWORD)v15 == 32 )
  {
    v19 = *(_QWORD *)(a3 + 88);
    v20 = *(_DWORD *)(v19 + 32);
    v21 = *(__int64 **)(v19 + 24);
    if ( v20 > 0x40 )
      v17 = *v21;
    else
      v17 = (__int64)((_QWORD)v21 << (64 - (unsigned __int8)v20)) >> (64 - (unsigned __int8)v20);
    v83.m128i_i64[0] = 1;
    v18 = a2[1];
    goto LABEL_9;
  }
  v16 = (_DWORD)v15 == 11 || (_DWORD)v15 == 33;
  if ( v16 )
  {
    v83.m128i_i64[0] = 3;
    v17 = *(_QWORD *)(a3 + 88);
    v18 = a2[1];
LABEL_9:
    v22 = *a2;
    v84 = 0;
    v85 = v17;
    return sub_1E1A9C0(v18, v22, &v83);
  }
  if ( v14 == 8 )
  {
    v24 = *(_DWORD *)(a3 + 84);
    v25 = (unsigned __int8 *)(*(_QWORD *)(a3 + 40) + 16LL * a4);
    v26 = *v25;
    if ( (_BYTE)v26 && (v27 = a1[4], (v28 = *(_QWORD *)(v27 + 8LL * (unsigned __int8)v26 + 120)) != 0) )
    {
      v29 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*(_QWORD *)v27 + 288LL);
      if ( v29 != sub_1D45FB0 )
      {
        v77 = a6;
        v81 = *(_DWORD *)(a3 + 84);
        v53 = ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int8 *, _QWORD))v29)(
                v27,
                v26,
                v15,
                v25,
                a10);
        a6 = v77;
        v24 = v81;
        v28 = v53;
      }
      if ( !a6 )
        goto LABEL_26;
      v71 = v24;
      v72 = (_QWORD *)v28;
      v78 = a6;
      v74 = (_QWORD *)a1[3];
      v30 = (_QWORD *)sub_1F3AD60(a1[2], a6, a5, v74, *a1);
      v31 = sub_1F4AAF0((__int64)v74, v30);
      v32 = v78;
      v24 = v71;
      if ( v72 && v31 && v72 != v31 && v71 < 0 )
      {
        v70 = v78;
        v80 = sub_1E6B9A0(a1[1], (__int64)v31, (unsigned __int8 *)byte_3F871B3, 0, (__int64)&v83, v78);
        v76 = a1[5];
        v45 = *(_QWORD *)(a1[2] + 8);
        v73 = (__int64 *)a1[6];
        v46 = *(_QWORD *)(v76 + 56);
        v47 = (__int64)sub_1E0B640(v46, v45 + 960, (__int64 *)(a3 + 72), 0);
        sub_1DD5BA0((__int64 *)(v76 + 16), v47);
        v48 = *v73;
        v49 = *(_QWORD *)v47 & 7LL;
        *(_QWORD *)(v47 + 8) = v73;
        v48 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v47 = v48 | v49;
        *(_QWORD *)(v48 + 8) = v47;
        *v73 = v47 | *v73 & 7;
        v83.m128i_i64[0] = 0x10000000;
        v83.m128i_i32[2] = v80;
        v84 = 0;
        v85 = 0;
        v86 = 0;
        sub_1E1A9C0(v47, v46, &v83);
        v83.m128i_i64[0] = 0;
        v83.m128i_i32[2] = v71;
        v84 = 0;
        v85 = 0;
        v86 = 0;
        sub_1E1A9C0(v47, v46, &v83);
        v24 = v80;
        v32 = v70;
      }
    }
    else
    {
      if ( !a6 )
      {
LABEL_26:
        v33 = a2[1];
        v34 = *a2;
        v83.m128i_i64[0] = 0;
        v84 = 0;
        *(__int32 *)((char *)v83.m128i_i32 + 3) = (unsigned __int8)(32 * v16);
        *(__int32 *)((char *)v83.m128i_i32 + 2) = v83.m128i_i16[1] & 0xF00F;
        v83.m128i_i32[2] = v24;
        v83.m128i_i32[0] &= 0xFFF000FF;
        v85 = 0;
        v86 = 0;
        return sub_1E1A9C0(v33, v34, &v83);
      }
      v40 = (_QWORD *)a1[3];
      v75 = v24;
      v79 = a6;
      v41 = (_QWORD *)sub_1F3AD60(a1[2], a6, a5, v40, *a1);
      sub_1F4AAF0((__int64)v40, v41);
      v32 = v79;
      v24 = v75;
    }
    if ( a5 >= *(unsigned __int16 *)(v32 + 2) )
      v16 = !(*(_QWORD *)(v32 + 8) & 1);
    goto LABEL_26;
  }
  if ( v14 == 9 )
  {
    v83.m128i_i64[0] = 12;
    v17 = *(_QWORD *)(a3 + 88);
    v18 = a2[1];
    goto LABEL_9;
  }
  if ( (unsigned __int16)(v14 - 34) <= 1u || (unsigned __int16)(v14 - 12) <= 1u )
  {
    v35 = *(_QWORD *)(a3 + 96);
    v36 = *(_QWORD *)(a3 + 88);
    v83.m128i_i8[0] = 10;
    v37 = a2[1];
LABEL_32:
    v83.m128i_i32[2] = v35;
    v84 = 0;
    v85 = v36;
    LODWORD(v86) = HIDWORD(v35);
    v38 = *(unsigned __int8 *)(a3 + 104);
LABEL_33:
    v39 = *a2;
    v83.m128i_i32[0] = (v38 << 8) | v83.m128i_i32[0] & 0xFFF000FF;
    return sub_1E1A9C0(v37, v39, &v83);
  }
  if ( v14 == 5 )
  {
    v42 = a2[1];
    v43 = *a2;
    v83.m128i_i8[0] = 4;
    v44 = *(_QWORD *)(a3 + 88);
    v84 = 0;
    v83.m128i_i32[0] &= 0xFFF000FF;
    v85 = v44;
    return sub_1E1A9C0(v42, v43, &v83);
  }
  if ( (_DWORD)v15 != 36 && (_DWORD)v15 != 14 )
  {
    switch ( (_DWORD)v15 )
    {
      case 0x25:
      case 0xF:
        v57 = *(_DWORD *)(a3 + 84);
        v37 = a2[1];
        v83.m128i_i8[0] = 8;
        v84 = 0;
        LODWORD(v85) = v57;
        v38 = *(unsigned __int8 *)(a3 + 88);
        goto LABEL_33;
      case 0x26:
      case 0x10:
        v82 = *(_DWORD *)(a3 + 100);
        v58 = *(_DWORD *)(a3 + 96) & 0x7FFFFFFF;
        v59 = sub_1D19C10(a3);
        v60 = v82;
        v61 = v59;
        if ( !v82 )
        {
          v66 = sub_1E0A0C0(*a1);
          v60 = sub_15AAE50(v66, v61);
          if ( !v60 )
          {
            v67 = sub_1E0A0C0(*a1);
            v60 = sub_12BE0A0(v67, v61);
          }
        }
        v62 = *(_QWORD *)(a3 + 88);
        v63 = *(_QWORD *)(*a1 + 64);
        if ( *(int *)(a3 + 96) < 0 )
          v64 = sub_1E0F850(v63, v62, v60);
        else
          v64 = sub_1E0DC70(v63, v62, v60);
        v83.m128i_i8[0] = 6;
        v37 = a2[1];
        v84 = 0;
        LODWORD(v85) = v64;
        v83.m128i_i32[2] = v58;
        LODWORD(v86) = 0;
        v38 = *(unsigned __int8 *)(a3 + 104);
        goto LABEL_33;
      case 0x11:
      case 0x27:
        v65 = *(_QWORD *)(a3 + 88);
        v83.m128i_i8[0] = 9;
        v84 = 0;
        v37 = a2[1];
        v85 = v65;
        v38 = *(unsigned __int8 *)(a3 + 96);
        v83.m128i_i32[2] = 0;
        LODWORD(v86) = 0;
        goto LABEL_33;
    }
    if ( v14 == 41 )
    {
      v50 = a2[1];
      v51 = *a2;
      v83.m128i_i8[0] = 15;
      v52 = *(_QWORD *)(a3 + 88);
      v84 = 0;
      v83.m128i_i32[0] &= 0xFFF000FF;
      v85 = v52;
      v83.m128i_i32[2] = 0;
      LODWORD(v86) = 0;
      return sub_1E1A9C0(v50, v51, &v83);
    }
    if ( (_DWORD)v15 == 40 || (_DWORD)v15 == 18 )
    {
      v35 = *(_QWORD *)(a3 + 96);
      v36 = *(_QWORD *)(a3 + 88);
      v83.m128i_i8[0] = 11;
      v37 = a2[1];
      goto LABEL_32;
    }
    if ( v14 == 42 )
    {
      v68 = *(_QWORD *)(a3 + 96);
      v37 = a2[1];
      v83.m128i_i8[0] = 7;
      v69 = *(_DWORD *)(a3 + 88);
      v84 = 0;
      LODWORD(v86) = HIDWORD(v68);
      v83.m128i_i32[2] = v68;
      v38 = *(unsigned __int8 *)(a3 + 84);
      LODWORD(v85) = v69;
      goto LABEL_33;
    }
    return sub_1FE67D0((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  }
  v54 = *(_DWORD *)(a3 + 84);
  v55 = a2[1];
  v83.m128i_i64[0] = 5;
  v56 = *a2;
  v84 = 0;
  LODWORD(v85) = v54;
  return sub_1E1A9C0(v55, v56, &v83);
}
