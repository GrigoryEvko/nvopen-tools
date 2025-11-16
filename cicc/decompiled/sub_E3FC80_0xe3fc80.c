// Function: sub_E3FC80
// Address: 0xe3fc80
//
__int64 __fastcall sub_E3FC80(__int64 a1, _BYTE *a2, size_t a3)
{
  _QWORD *v4; // rbx
  __int64 v5; // r12
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rcx
  __m128i *v10; // rax
  __int64 v11; // rcx
  __m128i *v12; // rdx
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rcx
  __m128i *v16; // rax
  __m128i **v17; // rcx
  __m128i *v18; // rdx
  __int64 v19; // rcx
  __m128i *v20; // rax
  __m128i *v21; // [rsp+0h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+8h] [rbp-B8h]
  __m128i v23; // [rsp+10h] [rbp-B0h] BYREF
  _QWORD *v24; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v25; // [rsp+28h] [rbp-98h]
  _QWORD v26[2]; // [rsp+30h] [rbp-90h] BYREF
  __m128i *v27; // [rsp+40h] [rbp-80h] BYREF
  __int64 v28; // [rsp+48h] [rbp-78h]
  __m128i v29; // [rsp+50h] [rbp-70h] BYREF
  __m128i **v30; // [rsp+60h] [rbp-60h] BYREF
  __int64 v31; // [rsp+68h] [rbp-58h]
  __m128i v32; // [rsp+70h] [rbp-50h] BYREF
  __int16 v33; // [rsp+80h] [rbp-40h]

  v4 = (_QWORD *)unk_4F8A318;
  if ( !unk_4F8A318 )
  {
LABEL_8:
    nullsub_392();
    if ( unk_4F8A318 )
    {
      if ( a2 )
      {
        v24 = v26;
        sub_E3FBA0((__int64 *)&v24, a2, (__int64)&a2[a3]);
      }
      else
      {
        LOBYTE(v26[0]) = 0;
        v24 = v26;
        v25 = 0;
      }
      v21 = &v23;
      sub_E3FBA0((__int64 *)&v21, "unsupported GC: ", (__int64)"");
      v7 = 15;
      v8 = 15;
      if ( v21 != &v23 )
        v8 = v23.m128i_i64[0];
      v9 = v22 + v25;
      if ( v22 + v25 <= v8 )
        goto LABEL_17;
      if ( v24 != v26 )
        v7 = v26[0];
      if ( v9 <= v7 )
      {
        v10 = (__m128i *)sub_2241130(&v24, 0, 0, v21, v22);
        v27 = &v29;
        v11 = v10->m128i_i64[0];
        v12 = v10 + 1;
        if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
          goto LABEL_18;
      }
      else
      {
LABEL_17:
        v10 = (__m128i *)sub_2241490(&v21, v24, v25, v9);
        v27 = &v29;
        v11 = v10->m128i_i64[0];
        v12 = v10 + 1;
        if ( (__m128i *)v10->m128i_i64[0] != &v10[1] )
        {
LABEL_18:
          v27 = (__m128i *)v11;
          v29.m128i_i64[0] = v10[1].m128i_i64[0];
          goto LABEL_19;
        }
      }
      v29 = _mm_loadu_si128(v10 + 1);
LABEL_19:
      v28 = v10->m128i_i64[1];
      v10->m128i_i64[0] = (__int64)v12;
      v10->m128i_i64[1] = 0;
      v10[1].m128i_i8[0] = 0;
      v33 = 260;
      v30 = &v27;
      sub_C64D30((__int64)&v30, 1u);
    }
    if ( a2 )
    {
      v27 = &v29;
      sub_E3FBA0((__int64 *)&v27, a2, (__int64)&a2[a3]);
    }
    else
    {
      v28 = 0;
      v27 = &v29;
      v29.m128i_i8[0] = 0;
    }
    v24 = v26;
    sub_E3FBA0((__int64 *)&v24, "unsupported GC: ", (__int64)"");
    v13 = 15;
    v14 = 15;
    if ( v24 != v26 )
      v14 = v26[0];
    v15 = v25 + v28;
    if ( v25 + v28 <= v14 )
      goto LABEL_29;
    if ( v27 != &v29 )
      v13 = v29.m128i_i64[0];
    if ( v15 <= v13 )
    {
      v16 = (__m128i *)sub_2241130(&v27, 0, 0, v24, v25);
      v30 = (__m128i **)&v32;
      v17 = (__m128i **)v16->m128i_i64[0];
      v18 = v16 + 1;
      if ( (__m128i *)v16->m128i_i64[0] != &v16[1] )
        goto LABEL_30;
    }
    else
    {
LABEL_29:
      v16 = (__m128i *)sub_2241490(&v24, v27, v28, v15);
      v30 = (__m128i **)&v32;
      v17 = (__m128i **)v16->m128i_i64[0];
      v18 = v16 + 1;
      if ( (__m128i *)v16->m128i_i64[0] != &v16[1] )
      {
LABEL_30:
        v30 = v17;
        v32.m128i_i64[0] = v16[1].m128i_i64[0];
        goto LABEL_31;
      }
    }
    v32 = _mm_loadu_si128(v16 + 1);
LABEL_31:
    v19 = v16->m128i_i64[1];
    v31 = v19;
    v16->m128i_i64[0] = (__int64)v18;
    v16->m128i_i64[1] = 0;
    v16[1].m128i_i8[0] = 0;
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v31) > 0x36 )
    {
      v20 = (__m128i *)sub_2241490(&v30, " (did you remember to link and initialize the library?)", 55, v19);
      v21 = &v23;
      if ( (__m128i *)v20->m128i_i64[0] == &v20[1] )
      {
        v23 = _mm_loadu_si128(v20 + 1);
      }
      else
      {
        v21 = (__m128i *)v20->m128i_i64[0];
        v23.m128i_i64[0] = v20[1].m128i_i64[0];
      }
      v22 = v20->m128i_i64[1];
      v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
      v20->m128i_i64[1] = 0;
      v20[1].m128i_i8[0] = 0;
      if ( v30 != (__m128i **)&v32 )
        j_j___libc_free_0(v30, v32.m128i_i64[0] + 1);
      if ( v24 != v26 )
        j_j___libc_free_0(v24, v26[0] + 1LL);
      if ( v27 != &v29 )
        j_j___libc_free_0(v27, v29.m128i_i64[0] + 1);
      v33 = 260;
      v30 = &v21;
      sub_C64D30((__int64)&v30, 1u);
    }
    sub_4262D8((__int64)"basic_string::append");
  }
  while ( 1 )
  {
    v5 = v4[1];
    if ( a3 == *(_QWORD *)(v5 + 8) && (!a3 || !memcmp(*(const void **)v5, a2, a3)) )
      break;
    v4 = (_QWORD *)*v4;
    if ( !v4 )
      goto LABEL_8;
  }
  (*(void (__fastcall **)(__int64))(v5 + 32))(a1);
  return a1;
}
