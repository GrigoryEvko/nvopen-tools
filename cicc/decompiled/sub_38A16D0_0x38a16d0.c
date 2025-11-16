// Function: sub_38A16D0
// Address: 0x38a16d0
//
__int64 __fastcall sub_38A16D0(__int64 a1, _QWORD *a2, double a3, double a4, double a5, __int64 a6, __int64 *a7)
{
  unsigned __int64 v8; // r15
  unsigned int v9; // eax
  unsigned int v10; // r12d
  __int64 *v11; // rax
  __m128i *v12; // rax
  __m128i *v13; // rax
  unsigned __int64 v14; // rdi
  __m128i *v16; // rax
  __m128i *v17; // rax
  __int64 *v18; // r14
  _QWORD *v19; // rax
  _QWORD *v20; // rbx
  __int64 *v21; // r14
  _QWORD *v22; // rax
  __int64 v23; // [rsp+0h] [rbp-F0h]
  __int64 v24; // [rsp+0h] [rbp-F0h]
  __int64 v25; // [rsp+10h] [rbp-E0h] BYREF
  _QWORD *v26; // [rsp+18h] [rbp-D8h] BYREF
  __int64 v27[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v28; // [rsp+30h] [rbp-C0h] BYREF
  const char **v29; // [rsp+40h] [rbp-B0h] BYREF
  __int64 v30; // [rsp+48h] [rbp-A8h]
  __m128i v31; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v32[2]; // [rsp+60h] [rbp-90h] BYREF
  __int64 v33; // [rsp+70h] [rbp-80h] BYREF
  const char **v34; // [rsp+80h] [rbp-70h] BYREF
  __int64 v35; // [rsp+88h] [rbp-68h]
  __m128i v36; // [rsp+90h] [rbp-60h] BYREF
  const char *v37; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-48h]
  _OWORD v39[4]; // [rsp+B0h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a1 + 56);
  v25 = 0;
  v37 = "expected type";
  LOWORD(v39[0]) = 259;
  v9 = sub_3891B00(a1, &v25, (__int64)&v37, 1);
  if ( (_BYTE)v9 )
    return 1;
  v10 = v9;
  v11 = *(__int64 **)(*(_QWORD *)(a7[1] + 24) + 16LL);
  if ( *(_BYTE *)(v25 + 8) )
  {
    v23 = *v11;
    v10 = sub_38A1070((__int64 **)a1, v25, &v26, a7, a3, a4, a5);
    if ( !(_BYTE)v10 )
    {
      if ( v23 != *v26 )
      {
        sub_3888960(v32, v23);
        v16 = (__m128i *)sub_2241130((unsigned __int64 *)v32, 0, 0, "value doesn't match function result type '", 0x2Au);
        v34 = (const char **)&v36;
        if ( (__m128i *)v16->m128i_i64[0] == &v16[1] )
        {
          v36 = _mm_loadu_si128(v16 + 1);
        }
        else
        {
          v34 = (const char **)v16->m128i_i64[0];
          v36.m128i_i64[0] = v16[1].m128i_i64[0];
        }
        v35 = v16->m128i_i64[1];
        v16->m128i_i64[0] = (__int64)v16[1].m128i_i64;
        v16->m128i_i64[1] = 0;
        v16[1].m128i_i8[0] = 0;
        if ( v35 != 0x3FFFFFFFFFFFFFFFLL )
        {
          v17 = (__m128i *)sub_2241490((unsigned __int64 *)&v34, "'", 1u);
          v37 = (const char *)v39;
          if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
          {
            v39[0] = _mm_loadu_si128(v17 + 1);
          }
          else
          {
            v37 = (const char *)v17->m128i_i64[0];
            *(_QWORD *)&v39[0] = v17[1].m128i_i64[0];
          }
          v38 = v17->m128i_i64[1];
          v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
          v17->m128i_i64[1] = 0;
          v17[1].m128i_i8[0] = 0;
          v31.m128i_i16[0] = 260;
          v29 = &v37;
          v10 = sub_38814C0(a1 + 8, v8, (__int64)&v29);
          if ( v37 != (const char *)v39 )
            j_j___libc_free_0((unsigned __int64)v37);
          if ( v34 != (const char **)&v36 )
            j_j___libc_free_0((unsigned __int64)v34);
          v14 = v32[0];
          if ( (__int64 *)v32[0] != &v33 )
            goto LABEL_14;
          return v10;
        }
LABEL_39:
        sub_4262D8((__int64)"basic_string::append");
      }
      v21 = *(__int64 **)a1;
      v24 = (__int64)v26;
      v22 = sub_1648A60(56, 1u);
      v20 = v22;
      if ( v22 )
        sub_15F6F90((__int64)v22, (__int64)v21, v24, 0);
LABEL_32:
      *a2 = v20;
      return v10;
    }
    return 1;
  }
  if ( !*(_BYTE *)(*v11 + 8) )
  {
    v18 = *(__int64 **)a1;
    v19 = sub_1648A60(56, 0);
    v20 = v19;
    if ( v19 )
      sub_15F6F90((__int64)v19, (__int64)v18, 0, 0);
    goto LABEL_32;
  }
  sub_3888960(v27, *v11);
  v12 = (__m128i *)sub_2241130((unsigned __int64 *)v27, 0, 0, "value doesn't match function result type '", 0x2Au);
  v29 = (const char **)&v31;
  if ( (__m128i *)v12->m128i_i64[0] == &v12[1] )
  {
    v31 = _mm_loadu_si128(v12 + 1);
  }
  else
  {
    v29 = (const char **)v12->m128i_i64[0];
    v31.m128i_i64[0] = v12[1].m128i_i64[0];
  }
  v30 = v12->m128i_i64[1];
  v12->m128i_i64[0] = (__int64)v12[1].m128i_i64;
  v12->m128i_i64[1] = 0;
  v12[1].m128i_i8[0] = 0;
  if ( v30 == 0x3FFFFFFFFFFFFFFFLL )
    goto LABEL_39;
  v13 = (__m128i *)sub_2241490((unsigned __int64 *)&v29, "'", 1u);
  v37 = (const char *)v39;
  if ( (__m128i *)v13->m128i_i64[0] == &v13[1] )
  {
    v39[0] = _mm_loadu_si128(v13 + 1);
  }
  else
  {
    v37 = (const char *)v13->m128i_i64[0];
    *(_QWORD *)&v39[0] = v13[1].m128i_i64[0];
  }
  v38 = v13->m128i_i64[1];
  v13->m128i_i64[0] = (__int64)v13[1].m128i_i64;
  v13->m128i_i64[1] = 0;
  v13[1].m128i_i8[0] = 0;
  v36.m128i_i16[0] = 260;
  v34 = &v37;
  v10 = sub_38814C0(a1 + 8, v8, (__int64)&v34);
  if ( v37 != (const char *)v39 )
    j_j___libc_free_0((unsigned __int64)v37);
  if ( v29 != (const char **)&v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v14 = v27[0];
  if ( (__int64 *)v27[0] != &v28 )
LABEL_14:
    j_j___libc_free_0(v14);
  return v10;
}
