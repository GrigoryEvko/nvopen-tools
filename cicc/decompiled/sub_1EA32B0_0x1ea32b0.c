// Function: sub_1EA32B0
// Address: 0x1ea32b0
//
__int64 __fastcall sub_1EA32B0(__int64 **a1, __int64 a2, __int64 *a3, __int32 a4, __int64 *a5, __int64 a6)
{
  __int64 *v6; // rbx
  __int64 result; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  unsigned __int8 *v11; // r15
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r13
  __int64 v15; // r15
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v22; // [rsp+28h] [rbp-A8h]
  __int64 v23; // [rsp+30h] [rbp-A0h]
  __int64 v24; // [rsp+40h] [rbp-90h]
  __int64 v25; // [rsp+48h] [rbp-88h]
  __int64 v26; // [rsp+50h] [rbp-80h]
  char v27; // [rsp+58h] [rbp-78h]
  __int64 v28[2]; // [rsp+60h] [rbp-70h] BYREF
  __m128i v29; // [rsp+70h] [rbp-60h] BYREF
  __int64 v30; // [rsp+80h] [rbp-50h]
  __int64 v31; // [rsp+88h] [rbp-48h]
  __int64 v32; // [rsp+90h] [rbp-40h]

  v6 = *a1;
  result = (__int64)&(*a1)[5 * *((unsigned int *)a1 + 2)];
  v23 = result;
  v22 = (__int64 *)(a2 + 16);
  if ( *a1 != (__int64 *)result )
  {
    while ( 1 )
    {
      v24 = *v6;
      v9 = v6[1];
      v25 = v9;
      v26 = v6[2];
      v27 = *((_BYTE *)v6 + 24);
      v10 = v6[4];
      v28[0] = v10;
      if ( v10 )
      {
        sub_1623A60((__int64)v28, v10, 2);
        v9 = v25;
      }
      if ( !*a5 )
        goto LABEL_13;
      v11 = sub_15B1000(*(unsigned __int8 **)(v9 - 8LL * *(unsigned int *)(v9 + 8)));
      v12 = sub_15C70A0((__int64)a5);
      if ( v11 != sub_15B1000(*(unsigned __int8 **)(v12 - 8LL * *(unsigned int *)(v12 + 8))) )
        break;
LABEL_15:
      v14 = *(_QWORD *)(a2 + 56);
      v15 = (__int64)sub_1E0B640(v14, *(_QWORD *)(a6 + 8) + 768LL, a5, 0);
      sub_1DD5BA0(v22, v15);
      v16 = *a3;
      *(_QWORD *)(v15 + 8) = a3;
      *(_QWORD *)v15 = v16 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v15 & 7LL;
      *(_QWORD *)((v16 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v15;
      v17 = *a3;
      v30 = 0;
      v31 = 0;
      *a3 = v15 | v17 & 7;
      v32 = 0;
      v29.m128i_i32[2] = a4;
      v29.m128i_i64[0] = 0x800000000LL;
      sub_1E1A9C0(v15, v14, &v29);
      if ( v27 )
      {
        v29.m128i_i64[0] = 1;
        v30 = 0;
        v31 = v24;
      }
      else
      {
        v29 = (__m128i)0x800000000uLL;
        v30 = 0;
        v31 = 0;
        v32 = 0;
      }
      sub_1E1A9C0(v15, v14, &v29);
      v29.m128i_i64[0] = 14;
      v30 = 0;
      v31 = v25;
      sub_1E1A9C0(v15, v14, &v29);
      v29.m128i_i64[0] = 14;
      v30 = 0;
      v31 = v26;
      result = sub_1E1A9C0(v15, v14, &v29);
      if ( v28[0] )
        result = sub_161E7C0((__int64)v28, v28[0]);
      v6 += 5;
      if ( (__int64 *)v23 == v6 )
        return result;
    }
    if ( *a5 )
      sub_161E7C0((__int64)a5, *a5);
LABEL_13:
    v13 = v28[0];
    *a5 = v28[0];
    if ( v13 )
      sub_1623A60((__int64)a5, v13, 2);
    goto LABEL_15;
  }
  return result;
}
