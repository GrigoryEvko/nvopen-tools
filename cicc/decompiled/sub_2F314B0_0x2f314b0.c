// Function: sub_2F314B0
// Address: 0x2f314b0
//
__int64 __fastcall sub_2F314B0(unsigned int *a1, __int64 a2, __int64 *a3, __int32 a4, unsigned __int8 **a5, __int64 a6)
{
  __int64 result; // rax
  __int64 *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int8 v13; // dl
  unsigned __int8 **v14; // rax
  unsigned __int8 *v15; // r12
  __int64 v16; // rax
  unsigned __int8 v17; // dl
  unsigned __int8 **v18; // rax
  bool v19; // zf
  unsigned __int8 *v20; // rsi
  __int64 v21; // rsi
  __int64 v22; // r15
  _QWORD *v23; // r12
  __int64 v24; // r15
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 *v28; // [rsp+8h] [rbp-A8h]
  __int64 *v32; // [rsp+28h] [rbp-88h]
  unsigned __int8 *v34; // [rsp+40h] [rbp-70h] BYREF
  unsigned __int8 *v35; // [rsp+48h] [rbp-68h] BYREF
  __m128i v36; // [rsp+50h] [rbp-60h] BYREF
  __int64 v37; // [rsp+60h] [rbp-50h]
  __int64 v38; // [rsp+68h] [rbp-48h]
  __int64 v39; // [rsp+70h] [rbp-40h]

  result = *(_QWORD *)a1 + 40LL * a1[2];
  v32 = (__int64 *)result;
  if ( *(_QWORD *)a1 != result )
  {
    v8 = (__int64 *)(*(_QWORD *)a1 + 32LL);
    v28 = (__int64 *)(a2 + 40);
    while ( 1 )
    {
      if ( *a5 )
      {
        v12 = *(v8 - 3);
        v13 = *(_BYTE *)(v12 - 16);
        if ( (v13 & 2) != 0 )
          v14 = *(unsigned __int8 ***)(v12 - 32);
        else
          v14 = (unsigned __int8 **)(v12 - 16 - 8LL * ((v13 >> 2) & 0xF));
        v15 = sub_AF34D0(*v14);
        v16 = sub_B10CD0((__int64)a5);
        v17 = *(_BYTE *)(v16 - 16);
        if ( (v17 & 2) != 0 )
          v18 = *(unsigned __int8 ***)(v16 - 32);
        else
          v18 = (unsigned __int8 **)(v16 - 16 - 8LL * ((v17 >> 2) & 0xF));
        v19 = v15 == sub_AF34D0(*v18);
        v20 = *a5;
        if ( v19 || a5 == (unsigned __int8 **)v8 )
          goto LABEL_17;
        if ( v20 )
          sub_B91220((__int64)a5, (__int64)v20);
      }
      else if ( a5 == (unsigned __int8 **)v8 )
      {
LABEL_34:
        v34 = 0;
        v22 = *(_QWORD *)(a6 + 8) - 560LL;
LABEL_35:
        v36.m128i_i64[0] = 0;
LABEL_36:
        v36.m128i_i64[1] = 0;
        v37 = 0;
        v35 = 0;
        v23 = *(_QWORD **)(a2 + 32);
        goto LABEL_21;
      }
      v21 = *v8;
      *a5 = (unsigned __int8 *)*v8;
      if ( !v21 )
        goto LABEL_34;
      sub_B96E90((__int64)a5, v21, 1);
      v20 = *a5;
LABEL_17:
      v34 = v20;
      v22 = *(_QWORD *)(a6 + 8) - 560LL;
      if ( !v20 )
        goto LABEL_35;
      sub_B96E90((__int64)&v34, (__int64)v20, 1);
      v36.m128i_i64[0] = (__int64)v34;
      if ( !v34 )
        goto LABEL_36;
      sub_B976B0((__int64)&v34, v34, (__int64)&v36);
      v34 = 0;
      v36.m128i_i64[1] = 0;
      v37 = 0;
      v23 = *(_QWORD **)(a2 + 32);
      v35 = (unsigned __int8 *)v36.m128i_i64[0];
      if ( v36.m128i_i64[0] )
        sub_B96E90((__int64)&v35, v36.m128i_i64[0], 1);
LABEL_21:
      v24 = (__int64)sub_2E7B380(v23, v22, &v35, 0);
      if ( v35 )
        sub_B91220((__int64)&v35, (__int64)v35);
      sub_2E31040(v28, v24);
      v25 = *a3;
      v26 = *(_QWORD *)v24;
      *(_QWORD *)(v24 + 8) = a3;
      v25 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v24 = v25 | v26 & 7;
      *(_QWORD *)(v25 + 8) = v24;
      v27 = v36.m128i_i64[1];
      *a3 = v24 | *a3 & 7;
      if ( v27 )
        sub_2E882B0(v24, (__int64)v23, v27);
      if ( v37 )
        sub_2E88680(v24, (__int64)v23, v37);
      if ( v36.m128i_i64[0] )
        sub_B91220((__int64)&v36, v36.m128i_i64[0]);
      if ( v34 )
        sub_B91220((__int64)&v34, (__int64)v34);
      v37 = 0;
      v36.m128i_i32[2] = a4;
      v38 = 0;
      v39 = 0;
      v36.m128i_i64[0] = 0x800000000LL;
      sub_2E8EAD0(v24, (__int64)v23, &v36);
      if ( *((_BYTE *)v8 - 8) )
      {
        v9 = *(v8 - 4);
        v36.m128i_i64[0] = 1;
        v37 = 0;
        v38 = v9;
      }
      else
      {
        v36 = (__m128i)0x800000000uLL;
        v37 = 0;
        v38 = 0;
        v39 = 0;
      }
      sub_2E8EAD0(v24, (__int64)v23, &v36);
      v10 = *(v8 - 3);
      v36.m128i_i64[0] = 14;
      v38 = v10;
      v37 = 0;
      sub_2E8EAD0(v24, (__int64)v23, &v36);
      v11 = *(v8 - 2);
      v36.m128i_i64[0] = 14;
      v38 = v11;
      v37 = 0;
      sub_2E8EAD0(v24, (__int64)v23, &v36);
      result = (__int64)(v8 + 5);
      if ( v32 == v8 + 1 )
        return result;
      v8 += 5;
    }
  }
  return result;
}
