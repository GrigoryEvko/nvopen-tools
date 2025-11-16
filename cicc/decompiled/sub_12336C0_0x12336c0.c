// Function: sub_12336C0
// Address: 0x12336c0
//
__int64 __fastcall sub_12336C0(__int64 a1, __int64 *a2, __int64 *a3, int a4)
{
  unsigned __int64 v6; // r12
  unsigned int v7; // r15d
  __m128i *v9; // rax
  __int64 v10; // rcx
  __m128i *v11; // rax
  __int64 v12; // rcx
  __m128i *v13; // r10
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rcx
  __m128i *v17; // rax
  __m128i *v18; // rcx
  __m128i *v19; // rdx
  __int64 v20; // rcx
  __m128i *v21; // rax
  __int64 v22; // rcx
  __int64 v23; // [rsp+10h] [rbp-130h] BYREF
  __int64 *v24; // [rsp+18h] [rbp-128h] BYREF
  __int64 v25[2]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v26; // [rsp+30h] [rbp-110h] BYREF
  __m128i *v27; // [rsp+40h] [rbp-100h] BYREF
  __int64 v28; // [rsp+48h] [rbp-F8h]
  __m128i v29; // [rsp+50h] [rbp-F0h] BYREF
  __m128i *v30; // [rsp+60h] [rbp-E0h] BYREF
  __int64 v31; // [rsp+68h] [rbp-D8h]
  __m128i v32; // [rsp+70h] [rbp-D0h] BYREF
  _QWORD *v33; // [rsp+80h] [rbp-C0h] BYREF
  __int64 v34; // [rsp+88h] [rbp-B8h]
  _QWORD v35[2]; // [rsp+90h] [rbp-B0h] BYREF
  __m128i *v36; // [rsp+A0h] [rbp-A0h] BYREF
  __int64 v37; // [rsp+A8h] [rbp-98h]
  __m128i v38; // [rsp+B0h] [rbp-90h] BYREF
  _QWORD v39[2]; // [rsp+C0h] [rbp-80h] BYREF
  __m128i v40; // [rsp+D0h] [rbp-70h] BYREF
  int v41[8]; // [rsp+E0h] [rbp-60h] BYREF
  __int16 v42; // [rsp+100h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 232);
  v24 = 0;
  if ( !(unsigned __int8)sub_122FE20((__int64 **)a1, &v23, a3)
    && !(unsigned __int8)sub_120AFE0(a1, 56, "expected 'to' after cast value") )
  {
    *(_QWORD *)v41 = "expected type";
    v42 = 259;
    v7 = sub_12190A0(a1, &v24, v41, 0);
    if ( !(_BYTE)v7 )
    {
      if ( (unsigned __int8)sub_B50F30(a4, *(_QWORD *)(v23 + 8), (__int64)v24) )
      {
        v42 = 257;
        *a2 = sub_B51D30(a4, v23, (__int64)v24, (__int64)v41, 0, 0);
        return v7;
      }
      sub_1207630((__int64 *)&v33, (__int64)v24);
      sub_1207630(v25, *(_QWORD *)(v23 + 8));
      v9 = (__m128i *)sub_2241130(v25, 0, 0, "invalid cast opcode for cast from '", 35);
      v27 = &v29;
      if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
      {
        v29 = _mm_loadu_si128(v9 + 1);
      }
      else
      {
        v27 = (__m128i *)v9->m128i_i64[0];
        v29.m128i_i64[0] = v9[1].m128i_i64[0];
      }
      v10 = v9->m128i_i64[1];
      v9[1].m128i_i8[0] = 0;
      v28 = v10;
      v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
      v9->m128i_i64[1] = 0;
      if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v28) <= 5 )
        goto LABEL_42;
      v11 = (__m128i *)sub_2241490(&v27, "' to '", 6, v10);
      v30 = &v32;
      if ( (__m128i *)v11->m128i_i64[0] == &v11[1] )
      {
        v32 = _mm_loadu_si128(v11 + 1);
      }
      else
      {
        v30 = (__m128i *)v11->m128i_i64[0];
        v32.m128i_i64[0] = v11[1].m128i_i64[0];
      }
      v12 = v11->m128i_i64[1];
      v11[1].m128i_i8[0] = 0;
      v31 = v12;
      v11->m128i_i64[0] = (__int64)v11[1].m128i_i64;
      v13 = v30;
      v11->m128i_i64[1] = 0;
      v14 = 15;
      v15 = 15;
      if ( v13 != &v32 )
        v15 = v32.m128i_i64[0];
      v16 = v31 + v34;
      if ( v31 + v34 <= v15 )
        goto LABEL_18;
      if ( v33 != v35 )
        v14 = v35[0];
      if ( v16 <= v14 )
      {
        v17 = (__m128i *)sub_2241130(&v33, 0, 0, v13, v31);
        v36 = &v38;
        v18 = (__m128i *)v17->m128i_i64[0];
        v19 = v17 + 1;
        if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
          goto LABEL_19;
      }
      else
      {
LABEL_18:
        v17 = (__m128i *)sub_2241490(&v30, v33, v34, v16);
        v36 = &v38;
        v18 = (__m128i *)v17->m128i_i64[0];
        v19 = v17 + 1;
        if ( (__m128i *)v17->m128i_i64[0] != &v17[1] )
        {
LABEL_19:
          v36 = v18;
          v38.m128i_i64[0] = v17[1].m128i_i64[0];
          goto LABEL_20;
        }
      }
      v38 = _mm_loadu_si128(v17 + 1);
LABEL_20:
      v20 = v17->m128i_i64[1];
      v37 = v20;
      v17->m128i_i64[0] = (__int64)v19;
      v17->m128i_i64[1] = 0;
      v17[1].m128i_i8[0] = 0;
      if ( v37 != 0x3FFFFFFFFFFFFFFFLL )
      {
        v21 = (__m128i *)sub_2241490(&v36, "'", 1, v20);
        v39[0] = &v40;
        if ( (__m128i *)v21->m128i_i64[0] == &v21[1] )
        {
          v40 = _mm_loadu_si128(v21 + 1);
        }
        else
        {
          v39[0] = v21->m128i_i64[0];
          v40.m128i_i64[0] = v21[1].m128i_i64[0];
        }
        v22 = v21->m128i_i64[1];
        v21[1].m128i_i8[0] = 0;
        v39[1] = v22;
        v21->m128i_i64[0] = (__int64)v21[1].m128i_i64;
        v21->m128i_i64[1] = 0;
        v42 = 260;
        *(_QWORD *)v41 = v39;
        sub_11FD800(a1 + 176, v6, (__int64)v41, 1);
        if ( (__m128i *)v39[0] != &v40 )
          j_j___libc_free_0(v39[0], v40.m128i_i64[0] + 1);
        if ( v36 != &v38 )
          j_j___libc_free_0(v36, v38.m128i_i64[0] + 1);
        if ( v30 != &v32 )
          j_j___libc_free_0(v30, v32.m128i_i64[0] + 1);
        if ( v27 != &v29 )
          j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
        if ( (__int64 *)v25[0] != &v26 )
          j_j___libc_free_0(v25[0], v26 + 1);
        if ( v33 != v35 )
          j_j___libc_free_0(v33, v35[0] + 1LL);
        return 1;
      }
LABEL_42:
      sub_4262D8((__int64)"basic_string::append");
    }
  }
  return 1;
}
