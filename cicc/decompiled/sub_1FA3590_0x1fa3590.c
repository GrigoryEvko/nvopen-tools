// Function: sub_1FA3590
// Address: 0x1fa3590
//
__int64 __fastcall sub_1FA3590(
        __int64 *a1,
        __int64 **a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        char a6,
        __m128 a7,
        double a8,
        __m128i a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        char a13,
        unsigned int a14)
{
  __int64 result; // rax
  bool v17; // al
  __int8 v18; // r11
  __int64 v19; // r8
  __int64 v20; // rax
  bool v21; // al
  __int8 v22; // cl
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // r11
  unsigned __int8 *v26; // rax
  __int64 v27; // r9
  __int64 v28; // r8
  __int64 v29; // rcx
  __m128i v30; // rax
  bool v31; // al
  __m128i v32; // xmm0
  bool v33; // bl
  __int64 v34; // r11
  bool v35; // al
  __int64 (*v36)(); // rax
  __int64 v37; // rsi
  const void **v38; // rbx
  __int64 v39; // rcx
  __int64 v40; // r14
  __int32 v41; // edx
  __int32 v42; // ebx
  __int64 v43; // [rsp+0h] [rbp-D0h]
  __int64 v44; // [rsp+8h] [rbp-C8h]
  __int64 v45; // [rsp+10h] [rbp-C0h]
  __int64 v46; // [rsp+18h] [rbp-B8h]
  __int64 v47; // [rsp+18h] [rbp-B8h]
  __int64 v48; // [rsp+18h] [rbp-B8h]
  __m128i v49; // [rsp+20h] [rbp-B0h] BYREF
  _BYTE **v50; // [rsp+30h] [rbp-A0h]
  __int64 *v51; // [rsp+38h] [rbp-98h]
  __int64 v52; // [rsp+40h] [rbp-90h] BYREF
  __int64 v53; // [rsp+48h] [rbp-88h]
  __m128i v54; // [rsp+50h] [rbp-80h] BYREF
  __int64 v55; // [rsp+60h] [rbp-70h]
  int v56; // [rsp+68h] [rbp-68h]
  __int64 *v57; // [rsp+70h] [rbp-60h] BYREF
  __int64 v58; // [rsp+78h] [rbp-58h]
  _BYTE v59[80]; // [rsp+80h] [rbp-50h] BYREF

  v52 = a4;
  v53 = a5;
  if ( *(_WORD *)(a11 + 24) != 185 || (*(_BYTE *)(a11 + 27) & 0xC) != 0 || (*(_WORD *)(a11 + 26) & 0x380) != 0 )
    return 0;
  if ( !a6 )
  {
    if ( !(_BYTE)v52 )
    {
      v51 = &v52;
      if ( !sub_1F58D20((__int64)&v52) )
      {
        v34 = (__int64)v51;
        if ( (*(_BYTE *)(a11 + 26) & 8) == 0 )
        {
          v50 = (_BYTE **)&v57;
          v51 = (__int64 *)v59;
          v57 = (__int64 *)v59;
          v49.m128i_i64[0] = v34;
          v58 = 0x400000000LL;
          if ( sub_1D18C00(a11, 1, a12) )
          {
            v22 = sub_1F58D20(v49.m128i_i64[0]);
            if ( !v22 )
              goto LABEL_11;
          }
          else
          {
            v48 = v49.m128i_i64[0];
            v49.m128i_i8[0] = sub_1F6D830(v52, v53, a10, a11, a12, a14, (__int64)v50, a3);
            v35 = sub_1F58D20(v48);
            v22 = v49.m128i_i8[0];
            if ( !v35 )
              goto LABEL_31;
          }
          goto LABEL_34;
        }
      }
      return 0;
    }
    if ( (unsigned __int8)(v52 - 14) > 0x5Fu && (*(_BYTE *)(a11 + 26) & 8) == 0 )
    {
      v50 = (_BYTE **)&v57;
      v51 = (__int64 *)v59;
      v57 = (__int64 *)v59;
      v49.m128i_i8[0] = v52;
      v58 = 0x400000000LL;
      v17 = sub_1D18C00(a11, 1, a12);
      v18 = v52;
      if ( v17 )
      {
LABEL_11:
        v19 = (unsigned int)a12;
LABEL_18:
        v23 = 16 * v19;
        v24 = *(_QWORD *)(a11 + 72);
        v25 = *(_QWORD *)(a11 + 104);
        v26 = (unsigned __int8 *)(*(_QWORD *)(a11 + 40) + v23);
        v45 = v23;
        v27 = *((_QWORD *)v26 + 1);
        v28 = *v26;
        v54.m128i_i64[0] = v24;
        v29 = *(_QWORD *)(a11 + 32);
        if ( v24 )
        {
          v43 = v28;
          v44 = v27;
          v46 = v25;
          v49.m128i_i64[0] = *(_QWORD *)(a11 + 32);
          sub_1623A60((__int64)&v54, v24, 2);
          v28 = v43;
          v27 = v44;
          v25 = v46;
          v29 = v49.m128i_i64[0];
        }
        v54.m128i_i32[2] = *(_DWORD *)(a11 + 64);
        v30.m128i_i64[0] = sub_1D2B590(
                             a1,
                             a13,
                             (__int64)&v54,
                             v52,
                             v53,
                             v25,
                             *(_OWORD *)v29,
                             *(_QWORD *)(v29 + 40),
                             *(_QWORD *)(v29 + 48),
                             v28,
                             v27);
        v49 = v30;
        v47 = v30.m128i_i64[0];
        if ( v54.m128i_i64[0] )
          sub_161E7C0((__int64)&v54, v54.m128i_i64[0]);
        sub_1FA0970(a2, (__int64)v50, a11, a12, v49.m128i_i64[0], v49.m128i_i64[1], a7, a8, a9, a14);
        v31 = sub_1D18C00(a11, 1, 0);
        v32 = _mm_load_si128(&v49);
        v33 = v31;
        v54 = v32;
        sub_1F994A0((__int64)a2, a10, v54.m128i_i64, 1, 1);
        if ( v33 )
        {
          sub_1D44C70((__int64)a1, a11, 1, v47, 1u);
        }
        else
        {
          v37 = *(_QWORD *)(a11 + 72);
          v38 = *(const void ***)(*(_QWORD *)(a11 + 40) + v45 + 8);
          v39 = *(unsigned __int8 *)(*(_QWORD *)(a11 + 40) + v45);
          v54.m128i_i64[0] = v37;
          if ( v37 )
          {
            v50 = (_BYTE **)v39;
            sub_1623A60((__int64)&v54, v37, 2);
            v39 = (__int64)v50;
          }
          v54.m128i_i32[2] = *(_DWORD *)(a11 + 64);
          v40 = sub_1D309E0(
                  a1,
                  145,
                  (__int64)&v54,
                  v39,
                  v38,
                  0,
                  *(double *)v32.m128i_i64,
                  a8,
                  *(double *)a9.m128i_i64,
                  *(_OWORD *)&v49);
          v42 = v41;
          if ( v54.m128i_i64[0] )
            sub_161E7C0((__int64)&v54, v54.m128i_i64[0]);
          v54.m128i_i64[0] = v40;
          v54.m128i_i32[2] = v42;
          v55 = v47;
          v56 = 1;
          sub_1F994A0((__int64)a2, a11, v54.m128i_i64, 2, 1);
        }
        result = a10;
        goto LABEL_25;
      }
LABEL_41:
      v49.m128i_i8[0] = v18;
      v22 = sub_1F6D830(v52, v53, a10, a11, a12, a14, (__int64)v50, a3);
      if ( (unsigned __int8)(v49.m128i_i8[0] - 14) > 0x5Fu )
        goto LABEL_31;
      goto LABEL_34;
    }
  }
  v20 = *(unsigned __int8 *)(*(_QWORD *)(a11 + 40) + 16LL * (unsigned int)a12);
  if ( !(_BYTE)v20
    || !(_BYTE)v52
    || (((int)*(unsigned __int16 *)(a3 + 2 * (v20 + 115LL * (unsigned __int8)v52 + 16104)) >> (4 * a13)) & 0xF) != 0 )
  {
    return 0;
  }
  v50 = (_BYTE **)&v57;
  v51 = (__int64 *)v59;
  v57 = (__int64 *)v59;
  v49.m128i_i8[0] = v52;
  v58 = 0x400000000LL;
  v21 = sub_1D18C00(a11, 1, a12);
  v18 = v52;
  v22 = v21;
  if ( !v21 )
    goto LABEL_41;
  if ( (unsigned __int8)(v52 - 14) > 0x5Fu )
  {
    v19 = (unsigned int)a12;
    goto LABEL_18;
  }
LABEL_34:
  v36 = *(__int64 (**)())(*(_QWORD *)a3 + 896LL);
  if ( v36 == sub_1F3CBC0 )
    goto LABEL_32;
  v49.m128i_i8[0] = v22;
  v22 &= ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v36)(a3, a10, 0);
LABEL_31:
  if ( v22 )
    goto LABEL_11;
LABEL_32:
  result = 0;
LABEL_25:
  if ( v57 != v51 )
  {
    v50 = 0;
    v51 = (__int64 *)result;
    _libc_free((unsigned __int64)v57);
    return (__int64)v51;
  }
  return result;
}
