// Function: sub_D31270
// Address: 0xd31270
//
__int64 __fastcall sub_D31270(
        __int64 a1,
        __int64 a2,
        unsigned __int8 a3,
        __int64 a4,
        _WORD *a5,
        int a6,
        _QWORD **a7,
        _BYTE *a8,
        _DWORD *a9)
{
  int v9; // eax
  _BYTE *v12; // r14
  unsigned __int8 *v13; // rax
  __int64 *v14; // r12
  __int64 v15; // r12
  unsigned __int8 *v16; // r13
  char v17; // bl
  __int64 v18; // rax
  unsigned __int8 *v19; // rax
  unsigned __int8 v20; // dl
  __m128i v21; // xmm4
  __m128i v22; // xmm2
  __m128i v24; // xmm0
  unsigned __int8 v25; // dl
  __int64 v26; // rax
  __int64 v27; // r13
  unsigned __int8 *v28; // r12
  unsigned __int8 *v29; // r12
  __m128i v30; // rax
  unsigned __int64 v31; // r12
  __m128i v32; // rax
  unsigned __int64 v33; // rax
  __int32 v34; // eax
  __int32 v35; // eax
  __int64 v36; // [rsp+0h] [rbp-160h]
  __int64 v37; // [rsp+0h] [rbp-160h]
  __int64 v38; // [rsp+28h] [rbp-138h]
  __int64 v39; // [rsp+38h] [rbp-128h]
  __int64 v40; // [rsp+48h] [rbp-118h]
  unsigned __int8 *v41; // [rsp+58h] [rbp-108h]
  unsigned __int8 v43; // [rsp+68h] [rbp-F8h]
  int v44; // [rsp+6Ch] [rbp-F4h]
  __int64 v45; // [rsp+70h] [rbp-F0h] BYREF
  unsigned int v46; // [rsp+78h] [rbp-E8h]
  const void *v47; // [rsp+80h] [rbp-E0h] BYREF
  unsigned int v48; // [rsp+88h] [rbp-D8h]
  const void *v49; // [rsp+90h] [rbp-D0h] BYREF
  unsigned int v50; // [rsp+98h] [rbp-C8h]
  __int64 v51; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned int v52; // [rsp+A8h] [rbp-B8h]
  __int64 v53; // [rsp+B0h] [rbp-B0h] BYREF
  unsigned int v54; // [rsp+B8h] [rbp-A8h]
  __int64 v55; // [rsp+C0h] [rbp-A0h]
  unsigned int v56; // [rsp+C8h] [rbp-98h]
  __int64 v57; // [rsp+D0h] [rbp-90h] BYREF
  unsigned int v58; // [rsp+D8h] [rbp-88h]
  __int64 v59; // [rsp+E0h] [rbp-80h]
  unsigned int v60; // [rsp+E8h] [rbp-78h]
  __m128i v61; // [rsp+F0h] [rbp-70h] BYREF
  __m128i v62; // [rsp+100h] [rbp-60h]
  __m128i v63; // [rsp+110h] [rbp-50h]
  char v64; // [rsp+120h] [rbp-40h]

  v9 = -1;
  if ( a6 )
    v9 = a6;
  v44 = v9;
  v12 = (_BYTE *)sub_AA4E30(a4);
  v13 = sub_BD3990(*(unsigned __int8 **)a1, a2);
  v14 = *(__int64 **)a5;
  v41 = v13;
  v43 = a3;
LABEL_4:
  if ( *(__int64 **)(a4 + 56) != v14 )
  {
    do
    {
      v15 = *v14;
      a5[4] = 0;
      v14 = (__int64 *)(v15 & 0xFFFFFFFFFFFFFFF8LL);
      v16 = (unsigned __int8 *)(v14 - 3);
      *(_QWORD *)a5 = v14;
      if ( !v14 )
        v16 = 0;
      v17 = sub_B46AA0((__int64)v16);
      if ( v17 )
        goto LABEL_4;
      *(_QWORD *)a5 = v14[1];
      if ( a9 )
        ++*a9;
      if ( !v44 )
        return 0;
      v18 = **(_QWORD **)a5;
      a5[4] = 0;
      *(_QWORD *)a5 = v18 & 0xFFFFFFFFFFFFFFF8LL;
      v40 = sub_D301A0((__int64)v16, v41, a2, v43, v12, a8);
      if ( v40 )
        return v40;
      if ( *v16 != 62 )
      {
        if ( !(unsigned __int8)sub_B46490((__int64)v16) )
          goto LABEL_18;
        if ( !a7 )
          goto LABEL_41;
        v24 = _mm_loadu_si128((const __m128i *)(a1 + 32));
        v61 = _mm_loadu_si128((const __m128i *)a1);
        v62 = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v63 = v24;
        goto LABEL_17;
      }
      v19 = sub_BD3990(*((unsigned __int8 **)v16 - 4), (__int64)v41);
      v20 = *v41;
      if ( *v41 <= 0x1Cu )
      {
        if ( v20 != 3 )
          goto LABEL_15;
      }
      else if ( v20 != 60 )
      {
        goto LABEL_15;
      }
      v25 = *v19;
      if ( *v19 > 0x1Cu )
      {
        if ( v25 != 60 )
        {
LABEL_15:
          if ( a7 )
            goto LABEL_16;
          goto LABEL_29;
        }
      }
      else if ( v25 != 3 )
      {
        goto LABEL_15;
      }
      if ( v41 != v19 )
        goto LABEL_18;
      if ( a7 )
      {
LABEL_16:
        v21 = _mm_loadu_si128((const __m128i *)(a1 + 32));
        v22 = _mm_loadu_si128((const __m128i *)(a1 + 16));
        v61 = _mm_loadu_si128((const __m128i *)a1);
        v62 = v22;
        v63 = v21;
LABEL_17:
        v64 = 1;
        if ( (sub_CF63E0(*a7, v16, &v61, (__int64)(a7 + 1)) & 2) != 0 )
          goto LABEL_41;
        goto LABEL_18;
      }
LABEL_29:
      v26 = *((_QWORD *)v16 - 8);
      v27 = *((_QWORD *)v16 - 4);
      v36 = *(_QWORD *)(v26 + 8);
      v28 = *(unsigned __int8 **)a1;
      v46 = sub_AE43F0((__int64)v12, *(_QWORD *)(*(_QWORD *)a1 + 8LL));
      if ( v46 > 0x40 )
        sub_C43690((__int64)&v45, 0, 0);
      else
        v45 = 0;
      v48 = sub_AE43F0((__int64)v12, *(_QWORD *)(v27 + 8));
      if ( v48 > 0x40 )
        sub_C43690((__int64)&v47, 0, 0);
      else
        v47 = 0;
      v29 = sub_BD45C0(v28, (__int64)v12, (__int64)&v45, 0, 0, 0, 0, v39);
      if ( v29 == sub_BD45C0((unsigned __int8 *)v27, (__int64)v12, (__int64)&v47, 0, 0, 0, 0, v38) )
      {
        v30.m128i_i64[0] = sub_9208B0((__int64)v12, a2);
        v61 = v30;
        v31 = (unsigned __int64)(v30.m128i_i64[0] + 7) >> 3;
        if ( v30.m128i_i8[8] )
          v31 |= 0x4000000000000000uLL;
        v32.m128i_i64[0] = sub_9208B0((__int64)v12, v36);
        v61 = v32;
        v33 = (unsigned __int64)(v32.m128i_i64[0] + 7) >> 3;
        v37 = v33;
        if ( v32.m128i_i8[8] )
          v37 = v33 | 0x4000000000000000LL;
        v58 = v46;
        if ( v46 > 0x40 )
          sub_C43780((__int64)&v57, (const void **)&v45);
        else
          v57 = v45;
        sub_C46A40((__int64)&v57, v31);
        v34 = v58;
        v58 = 0;
        v61.m128i_i32[2] = v34;
        v61.m128i_i64[0] = v57;
        v52 = v46;
        if ( v46 > 0x40 )
          sub_C43780((__int64)&v51, (const void **)&v45);
        else
          v51 = v45;
        sub_AADC30((__int64)&v53, (__int64)&v51, v61.m128i_i64);
        if ( v52 > 0x40 && v51 )
          j_j___libc_free_0_0(v51);
        if ( v61.m128i_i32[2] > 0x40u && v61.m128i_i64[0] )
          j_j___libc_free_0_0(v61.m128i_i64[0]);
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        v52 = v48;
        if ( v48 > 0x40 )
          sub_C43780((__int64)&v51, &v47);
        else
          v51 = (__int64)v47;
        sub_C46A40((__int64)&v51, v37);
        v35 = v52;
        v52 = 0;
        v61.m128i_i32[2] = v35;
        v61.m128i_i64[0] = v51;
        v50 = v48;
        if ( v48 > 0x40 )
          sub_C43780((__int64)&v49, &v47);
        else
          v49 = v47;
        sub_AADC30((__int64)&v57, (__int64)&v49, v61.m128i_i64);
        if ( v50 > 0x40 && v49 )
          j_j___libc_free_0_0(v49);
        if ( v61.m128i_i32[2] > 0x40u && v61.m128i_i64[0] )
          j_j___libc_free_0_0(v61.m128i_i64[0]);
        if ( v52 > 0x40 && v51 )
          j_j___libc_free_0_0(v51);
        sub_AB2160((__int64)&v61, (__int64)&v53, (__int64)&v57, 0);
        v17 = sub_AAF7D0((__int64)&v61);
        if ( v62.m128i_i32[2] > 0x40u && v62.m128i_i64[0] )
          j_j___libc_free_0_0(v62.m128i_i64[0]);
        if ( v61.m128i_i32[2] > 0x40u && v61.m128i_i64[0] )
          j_j___libc_free_0_0(v61.m128i_i64[0]);
        if ( v60 > 0x40 && v59 )
          j_j___libc_free_0_0(v59);
        if ( v58 > 0x40 && v57 )
          j_j___libc_free_0_0(v57);
        if ( v56 > 0x40 && v55 )
          j_j___libc_free_0_0(v55);
        if ( v54 > 0x40 && v53 )
          j_j___libc_free_0_0(v53);
      }
      if ( v48 > 0x40 && v47 )
        j_j___libc_free_0_0(v47);
      if ( v46 > 0x40 && v45 )
        j_j___libc_free_0_0(v45);
      if ( !v17 )
      {
LABEL_41:
        *(_QWORD *)a5 = *(_QWORD *)(*(_QWORD *)a5 + 8LL);
        a5[4] = 0;
        return v40;
      }
LABEL_18:
      --v44;
      v14 = *(__int64 **)a5;
    }
    while ( *(_QWORD *)(a4 + 56) != *(_QWORD *)a5 );
  }
  return 0;
}
