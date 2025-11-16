// Function: sub_37B9D40
// Address: 0x37b9d40
//
_QWORD *__fastcall sub_37B9D40(_QWORD *a1, const __m128i *a2, __int64 *a3, __int64 *a4)
{
  __int64 v6; // r8
  __int64 v7; // rdx
  unsigned __int8 v8; // al
  __int64 *v9; // rcx
  __int64 v10; // rcx
  __int64 *v11; // rdi
  _QWORD *v12; // rax
  __int64 v13; // rax
  _QWORD *v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // r12
  __int64 v17; // rax
  __int64 v19; // [rsp+10h] [rbp-90h]
  unsigned __int8 *v21; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int8 *v22; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int8 *v23; // [rsp+38h] [rbp-68h] BYREF
  __m128i v24; // [rsp+40h] [rbp-60h] BYREF
  __int64 v25; // [rsp+50h] [rbp-50h]
  __int64 v26; // [rsp+58h] [rbp-48h]
  __int64 v27; // [rsp+60h] [rbp-40h]

  v6 = a3[4];
  v7 = *a3;
  v8 = *(_BYTE *)(v7 - 16);
  if ( (v8 & 2) != 0 )
    v9 = *(__int64 **)(v7 - 32);
  else
    v9 = (__int64 *)(v7 - 16 - 8LL * ((v8 >> 2) & 0xF));
  v10 = *v9;
  v11 = (__int64 *)(*(_QWORD *)(v7 + 8) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(v7 + 8) & 4) != 0 )
    v11 = (__int64 *)*v11;
  v12 = sub_B01860(v11, 0, 0, v10, v6, 0, 0, 1);
  sub_B10CB0(&v21, (__int64)v12);
  v13 = *(_QWORD *)(*a1 + 8LL);
  v22 = v21;
  v19 = v13 - 560;
  if ( !v21 )
  {
    v24.m128i_i64[0] = 0;
    goto LABEL_27;
  }
  sub_B96E90((__int64)&v22, (__int64)v21, 1);
  v24.m128i_i64[0] = (__int64)v22;
  if ( !v22 )
  {
LABEL_27:
    v24.m128i_i64[1] = 0;
    v14 = (_QWORD *)a1[3];
    v25 = 0;
    v23 = 0;
    goto LABEL_9;
  }
  sub_B976B0((__int64)&v22, v22, (__int64)&v24);
  v22 = 0;
  v24.m128i_i64[1] = 0;
  v14 = (_QWORD *)a1[3];
  v25 = 0;
  v23 = (unsigned __int8 *)v24.m128i_i64[0];
  if ( v24.m128i_i64[0] )
    sub_B96E90((__int64)&v23, v24.m128i_i64[0], 1);
LABEL_9:
  v15 = sub_2E7B380(v14, v19, &v23, 0);
  v16 = (__int64)v15;
  if ( v24.m128i_i64[1] )
    sub_2E882B0((__int64)v15, (__int64)v14, v24.m128i_i64[1]);
  if ( v25 )
    sub_2E88680(v16, (__int64)v14, v25);
  if ( v23 )
    sub_B91220((__int64)&v23, (__int64)v23);
  if ( v24.m128i_i64[0] )
    sub_B91220((__int64)&v24, v24.m128i_i64[0]);
  if ( v22 )
    sub_B91220((__int64)&v22, (__int64)v22);
  sub_2E8EAD0(v16, (__int64)v14, a2);
  if ( *((_BYTE *)a4 + 8) )
  {
    v24.m128i_i64[0] = 1;
    v25 = 0;
    v26 = 0;
  }
  else
  {
    v24 = 0u;
    v25 = 0;
    v26 = 0;
    v27 = 0;
  }
  sub_2E8EAD0(v16, (__int64)v14, &v24);
  v17 = *a3;
  v24.m128i_i64[0] = 14;
  v26 = v17;
  v25 = 0;
  sub_2E8EAD0(v16, (__int64)v14, &v24);
  v24.m128i_i64[0] = 14;
  v25 = 0;
  v26 = *a4;
  sub_2E8EAD0(v16, (__int64)v14, &v24);
  if ( v21 )
    sub_B91220((__int64)&v21, (__int64)v21);
  return v14;
}
