// Function: sub_228A0E0
// Address: 0x228a0e0
//
void __fastcall sub_228A0E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r9
  __m128i *v5; // rax
  unsigned __int64 v6; // rcx
  _BYTE *v7; // rdi
  _QWORD *v8; // [rsp+8h] [rbp-1A8h] BYREF
  unsigned __int64 v9[2]; // [rsp+10h] [rbp-1A0h] BYREF
  __m128i v10; // [rsp+20h] [rbp-190h] BYREF
  _BYTE *v11; // [rsp+30h] [rbp-180h] BYREF
  __int64 v12; // [rsp+38h] [rbp-178h]
  _QWORD v13[2]; // [rsp+40h] [rbp-170h] BYREF
  unsigned __int64 v14[2]; // [rsp+50h] [rbp-160h] BYREF
  _BYTE v15[16]; // [rsp+60h] [rbp-150h] BYREF
  __int64 *v16; // [rsp+70h] [rbp-140h] BYREF
  __int16 v17; // [rsp+90h] [rbp-120h]
  void *v18[2]; // [rsp+A0h] [rbp-110h] BYREF
  _QWORD v19[2]; // [rsp+B0h] [rbp-100h] BYREF
  __int16 v20; // [rsp+C0h] [rbp-F0h]
  _BYTE v21[80]; // [rsp+D0h] [rbp-E0h] BYREF
  _QWORD v22[5]; // [rsp+120h] [rbp-90h] BYREF
  unsigned int v23; // [rsp+148h] [rbp-68h]
  _BYTE v24[16]; // [rsp+158h] [rbp-58h] BYREF
  void (__fastcall *v25)(_BYTE *, _BYTE *, __int64); // [rsp+168h] [rbp-48h]

  sub_D12090((__int64)v21, a1);
  sub_2286A70((__int64)v22, a1, (__int64)v21, a2, a3, v4);
  v18[0] = v19;
  sub_11F4570((__int64 *)v18, *(_BYTE **)(v22[0] + 168LL), *(_QWORD *)(v22[0] + 168LL) + *(_QWORD *)(v22[0] + 176LL));
  v5 = (__m128i *)sub_2241130((unsigned __int64 *)v18, 0, 0, "Call graph: ", 0xCu);
  v9[0] = (unsigned __int64)&v10;
  if ( (__m128i *)v5->m128i_i64[0] == &v5[1] )
  {
    v10 = _mm_loadu_si128(v5 + 1);
  }
  else
  {
    v9[0] = v5->m128i_i64[0];
    v10.m128i_i64[0] = v5[1].m128i_i64[0];
  }
  v6 = v5->m128i_u64[1];
  v5[1].m128i_i8[0] = 0;
  v9[1] = v6;
  v5->m128i_i64[0] = (__int64)v5[1].m128i_i64;
  v5->m128i_i64[1] = 0;
  if ( v18[0] != v19 )
    j_j___libc_free_0((unsigned __int64)v18[0]);
  v20 = 260;
  v18[0] = v9;
  v17 = 260;
  v16 = &qword_4FDAF48;
  v8 = v22;
  v14[0] = (unsigned __int64)v15;
  v14[1] = 0;
  v15[0] = 0;
  sub_2289AE0((__int64)&v11, (__int64)&v8, (void **)&v16, 1, v18, (__int64)v14);
  if ( (_BYTE *)v14[0] != v15 )
    j_j___libc_free_0(v14[0]);
  v7 = v11;
  if ( v12 )
  {
    sub_C67930(v11, v12, 0, 0);
    v7 = v11;
    if ( v11 == (_BYTE *)v13 )
      goto LABEL_10;
    goto LABEL_9;
  }
  if ( v11 != (_BYTE *)v13 )
LABEL_9:
    j_j___libc_free_0((unsigned __int64)v7);
LABEL_10:
  if ( (__m128i *)v9[0] != &v10 )
    j_j___libc_free_0(v9[0]);
  if ( v25 )
    v25(v24, v24, 3);
  sub_C7D6A0(v22[3], 16LL * v23, 8);
  sub_D0FA70((__int64)v21);
}
