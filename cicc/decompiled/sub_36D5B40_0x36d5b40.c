// Function: sub_36D5B40
// Address: 0x36d5b40
//
__int64 __fastcall sub_36D5B40(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        const __m128i *a5,
        __int64 a6,
        unsigned __int8 **a7)
{
  __int64 v8; // r15
  unsigned __int8 *v9; // rsi
  __int64 v10; // r15
  _QWORD *v11; // r14
  __int64 v12; // r13
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // r13
  unsigned __int8 *v16; // rsi
  _QWORD *v17; // r14
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v22; // r15
  _QWORD *v23; // r14
  _QWORD *v24; // r13
  __int64 v25; // r15
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r15
  _QWORD *v30; // r14
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  _QWORD *v37; // [rsp+28h] [rbp-98h]
  unsigned __int8 *v38; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int8 *v39; // [rsp+40h] [rbp-80h] BYREF
  __int64 v40; // [rsp+48h] [rbp-78h]
  __int64 v41; // [rsp+50h] [rbp-70h]
  __m128i v42; // [rsp+60h] [rbp-60h] BYREF
  __int64 v43; // [rsp+70h] [rbp-50h]
  __int64 v44; // [rsp+78h] [rbp-48h]

  v8 = *(_QWORD *)(a1 + 8);
  v9 = *a7;
  if ( a4 )
  {
    v38 = *a7;
    v10 = v8 - 15840;
    if ( v9 )
    {
      sub_B96E90((__int64)&v38, (__int64)v9, 1);
      v39 = v38;
      if ( v38 )
      {
        sub_B976B0((__int64)&v38, v38, (__int64)&v39);
        v38 = 0;
        v40 = 0;
        v11 = (_QWORD *)a2[4];
        v41 = 0;
        v37 = a2 + 6;
        v42.m128i_i64[0] = (__int64)v39;
        if ( v39 )
          sub_B96E90((__int64)&v42, (__int64)v39, 1);
LABEL_6:
        v12 = (__int64)sub_2E7B380(v11, v10, (unsigned __int8 **)&v42, 0);
        if ( v42.m128i_i64[0] )
          sub_B91220((__int64)&v42, v42.m128i_i64[0]);
        sub_2E31040(a2 + 5, v12);
        v13 = a2[6];
        *(_QWORD *)(v12 + 8) = v37;
        v13 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v12 = v13 | *(_QWORD *)v12 & 7LL;
        *(_QWORD *)(v13 + 8) = v12;
        v14 = v40;
        a2[6] = v12 | a2[6] & 7LL;
        if ( v14 )
          sub_2E882B0(v12, (__int64)v11, v14);
        if ( v41 )
          sub_2E88680(v12, (__int64)v11, v41);
        sub_2E8EAD0(v12, (__int64)v11, a5);
        v42.m128i_i8[0] = 4;
        v43 = 0;
        v42.m128i_i32[0] &= 0xFFF000FF;
        v44 = a3;
        sub_2E8EAD0(v12, (__int64)v11, &v42);
        if ( v39 )
          sub_B91220((__int64)&v39, (__int64)v39);
        if ( v38 )
          sub_B91220((__int64)&v38, (__int64)v38);
        v15 = *(_QWORD *)(a1 + 8) - 60520LL;
        v16 = *a7;
        v38 = v16;
        if ( v16 )
        {
          sub_B96E90((__int64)&v38, (__int64)v16, 1);
          v39 = v38;
          if ( v38 )
          {
            sub_B976B0((__int64)&v38, v38, (__int64)&v39);
            v17 = (_QWORD *)a2[4];
            v38 = 0;
            v40 = 0;
            v41 = 0;
            v42.m128i_i64[0] = (__int64)v39;
            if ( v39 )
              sub_B96E90((__int64)&v42, (__int64)v39, 1);
            goto LABEL_20;
          }
        }
        else
        {
          v39 = 0;
        }
        v40 = 0;
        v17 = (_QWORD *)a2[4];
        v41 = 0;
        v42.m128i_i64[0] = 0;
LABEL_20:
        v18 = (__int64)sub_2E7B380(v17, v15, (unsigned __int8 **)&v42, 0);
        if ( v42.m128i_i64[0] )
          sub_B91220((__int64)&v42, v42.m128i_i64[0]);
        sub_2E31040(a2 + 5, v18);
        v19 = a2[6];
        *(_QWORD *)(v18 + 8) = v37;
        v19 &= 0xFFFFFFFFFFFFFFF8LL;
        *(_QWORD *)v18 = v19 | *(_QWORD *)v18 & 7LL;
        *(_QWORD *)(v19 + 8) = v18;
        v20 = v40;
        a2[6] = v18 | a2[6] & 7LL;
        if ( v20 )
          sub_2E882B0(v18, (__int64)v17, v20);
        if ( v41 )
          sub_2E88680(v18, (__int64)v17, v41);
        v42.m128i_i8[0] = 4;
        v43 = 0;
        v42.m128i_i32[0] &= 0xFFF000FF;
        v44 = a4;
        sub_2E8EAD0(v18, (__int64)v17, &v42);
        if ( v39 )
          sub_B91220((__int64)&v39, (__int64)v39);
        if ( v38 )
          sub_B91220((__int64)&v38, (__int64)v38);
        return 2;
      }
    }
    else
    {
      v39 = 0;
    }
    v11 = (_QWORD *)a2[4];
    v40 = 0;
    v41 = 0;
    v37 = a2 + 6;
    v42.m128i_i64[0] = 0;
    goto LABEL_6;
  }
  if ( !a6 )
  {
    v38 = *a7;
    v29 = v8 - 60520;
    if ( v9 )
    {
      sub_B96E90((__int64)&v38, (__int64)v9, 1);
      v39 = v38;
      if ( v38 )
      {
        v30 = a2 + 6;
        sub_B976B0((__int64)&v38, v38, (__int64)&v39);
        v38 = 0;
        v40 = 0;
        v24 = (_QWORD *)a2[4];
        v41 = 0;
        v42.m128i_i64[0] = (__int64)v39;
        if ( v39 )
          sub_B96E90((__int64)&v42, (__int64)v39, 1);
        goto LABEL_56;
      }
    }
    else
    {
      v39 = 0;
    }
    v40 = 0;
    v24 = (_QWORD *)a2[4];
    v30 = a2 + 6;
    v41 = 0;
    v42.m128i_i64[0] = 0;
LABEL_56:
    v25 = (__int64)sub_2E7B380(v24, v29, (unsigned __int8 **)&v42, 0);
    if ( v42.m128i_i64[0] )
      sub_B91220((__int64)&v42, v42.m128i_i64[0]);
    sub_2E31040(a2 + 5, v25);
    v31 = a2[6];
    v32 = *(_QWORD *)v25;
    *(_QWORD *)(v25 + 8) = v30;
    v31 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v25 = v31 | v32 & 7;
    *(_QWORD *)(v31 + 8) = v25;
    v33 = v40;
    a2[6] = v25 | a2[6] & 7LL;
    if ( v33 )
      sub_2E882B0(v25, (__int64)v24, v33);
    if ( v41 )
      sub_2E88680(v25, (__int64)v24, v41);
    goto LABEL_47;
  }
  v38 = *a7;
  v22 = v8 - 15840;
  if ( v9 )
  {
    sub_B96E90((__int64)&v38, (__int64)v9, 1);
    v39 = v38;
    if ( v38 )
    {
      v23 = a2 + 6;
      sub_B976B0((__int64)&v38, v38, (__int64)&v39);
      v38 = 0;
      v40 = 0;
      v24 = (_QWORD *)a2[4];
      v41 = 0;
      v42.m128i_i64[0] = (__int64)v39;
      if ( v39 )
        sub_B96E90((__int64)&v42, (__int64)v39, 1);
      goto LABEL_40;
    }
  }
  else
  {
    v39 = 0;
  }
  v40 = 0;
  v24 = (_QWORD *)a2[4];
  v23 = a2 + 6;
  v41 = 0;
  v42.m128i_i64[0] = 0;
LABEL_40:
  v25 = (__int64)sub_2E7B380(v24, v22, (unsigned __int8 **)&v42, 0);
  if ( v42.m128i_i64[0] )
    sub_B91220((__int64)&v42, v42.m128i_i64[0]);
  sub_2E31040(a2 + 5, v25);
  v26 = a2[6];
  v27 = *(_QWORD *)v25;
  *(_QWORD *)(v25 + 8) = v23;
  v26 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v25 = v26 | v27 & 7;
  *(_QWORD *)(v26 + 8) = v25;
  v28 = v40;
  a2[6] = v25 | a2[6] & 7LL;
  if ( v28 )
    sub_2E882B0(v25, (__int64)v24, v28);
  if ( v41 )
    sub_2E88680(v25, (__int64)v24, v41);
  sub_2E8EAD0(v25, (__int64)v24, a5);
LABEL_47:
  v42.m128i_i8[0] = 4;
  v43 = 0;
  v42.m128i_i32[0] &= 0xFFF000FF;
  v44 = a3;
  sub_2E8EAD0(v25, (__int64)v24, &v42);
  if ( v39 )
    sub_B91220((__int64)&v39, (__int64)v39);
  if ( v38 )
    sub_B91220((__int64)&v38, (__int64)v38);
  return 1;
}
