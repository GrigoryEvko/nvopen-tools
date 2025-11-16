// Function: sub_387DD30
// Address: 0x387dd30
//
__int64 __fastcall sub_387DD30(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __m128i a4,
        __m128i a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        __m128 a11)
{
  _QWORD *v13; // rax
  __int64 **v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  double v18; // xmm4_8
  double v19; // xmm5_8
  __int64 *v20; // r14
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  unsigned __int8 *v24; // rsi
  __int64 v25; // rax
  _QWORD *v26; // r13
  __int64 v27; // rsi
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 *v30; // r13
  __int64 v31; // rax
  __int64 v32; // rcx
  __int64 v33; // rsi
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int8 *v36; // rsi
  __int64 *v38; // [rsp+18h] [rbp-98h]
  __int64 v39; // [rsp+20h] [rbp-90h]
  unsigned __int8 *v40; // [rsp+38h] [rbp-78h] BYREF
  __int64 v41; // [rsp+40h] [rbp-70h] BYREF
  __int16 v42; // [rsp+50h] [rbp-60h]
  _QWORD v43[2]; // [rsp+60h] [rbp-50h] BYREF
  __int16 v44; // [rsp+70h] [rbp-40h]

  v13 = (_QWORD *)sub_16498A0(a3);
  v14 = (__int64 **)sub_1644900(v13, 1u);
  v17 = sub_15A06D0(v14, 1, v15, v16);
  v20 = *(__int64 **)(a2 + 40);
  v21 = v17;
  v38 = &v20[*(unsigned int *)(a2 + 48)];
  if ( v38 != v20 )
  {
    v39 = (__int64)(a1 + 33);
    while ( 1 )
    {
      v26 = sub_387DD00(a1, *v20, a3, a4, a5, a6, a7, v18, v19, a10, a11);
      a1[34] = *(_QWORD *)(a3 + 40);
      a1[35] = a3 + 24;
      v27 = *(_QWORD *)(a3 + 48);
      v43[0] = v27;
      if ( v27 )
        break;
      v24 = (unsigned __int8 *)a1[33];
      if ( v24 )
        goto LABEL_4;
      v42 = 257;
      if ( *((_BYTE *)v26 + 16) > 0x10u )
        goto LABEL_16;
LABEL_8:
      if ( sub_1593BB0((__int64)v26, (__int64)v24, v22, v23) )
      {
LABEL_12:
        if ( v38 == ++v20 )
          return v21;
      }
      else
      {
        if ( *(_BYTE *)(v21 + 16) <= 0x10u )
        {
          v21 = sub_15A2D10((__int64 *)v21, (__int64)v26, *(double *)a4.m128i_i64, *(double *)a5.m128i_i64, a6);
          v25 = sub_14DBA30(v21, a1[41], 0);
          if ( v25 )
            v21 = v25;
          goto LABEL_12;
        }
LABEL_16:
        v44 = 257;
        v28 = sub_15FB440(27, (__int64 *)v21, (__int64)v26, (__int64)v43, 0);
        v29 = a1[34];
        v21 = v28;
        if ( v29 )
        {
          v30 = (__int64 *)a1[35];
          sub_157E9D0(v29 + 40, v28);
          v31 = *(_QWORD *)(v21 + 24);
          v32 = *v30;
          *(_QWORD *)(v21 + 32) = v30;
          v32 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)(v21 + 24) = v32 | v31 & 7;
          *(_QWORD *)(v32 + 8) = v21 + 24;
          *v30 = *v30 & 7 | (v21 + 24);
        }
        sub_164B780(v21, &v41);
        v33 = a1[33];
        if ( !v33 )
          goto LABEL_12;
        v40 = (unsigned __int8 *)a1[33];
        sub_1623A60((__int64)&v40, v33, 2);
        v34 = *(_QWORD *)(v21 + 48);
        v35 = v21 + 48;
        if ( v34 )
        {
          sub_161E7C0(v21 + 48, v34);
          v35 = v21 + 48;
        }
        v36 = v40;
        *(_QWORD *)(v21 + 48) = v40;
        if ( !v36 )
          goto LABEL_12;
        ++v20;
        sub_1623210((__int64)&v40, v36, v35);
        if ( v38 == v20 )
          return v21;
      }
    }
    sub_1623A60((__int64)v43, v27, 2);
    v24 = (unsigned __int8 *)a1[33];
    if ( v24 )
LABEL_4:
      sub_161E7C0(v39, (__int64)v24);
    v24 = (unsigned __int8 *)v43[0];
    a1[33] = v43[0];
    if ( v24 )
      sub_1623210((__int64)v43, v24, v39);
    v42 = 257;
    if ( *((_BYTE *)v26 + 16) > 0x10u )
      goto LABEL_16;
    goto LABEL_8;
  }
  return v21;
}
