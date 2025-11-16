// Function: sub_26ECF60
// Address: 0x26ecf60
//
void __fastcall sub_26ECF60(__int64 a1, _BYTE *a2, __int64 a3, _QWORD *a4)
{
  __m128i *v6; // rax
  __int64 v7; // rcx
  char *v8; // rsi
  __m128i *v9; // rax
  size_t v10; // rcx
  _OWORD *v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rax
  __m128i *v14; // [rsp+0h] [rbp-90h]
  size_t v15; // [rsp+8h] [rbp-88h]
  __m128i v16; // [rsp+10h] [rbp-80h] BYREF
  __int64 v17[2]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v18[2]; // [rsp+30h] [rbp-60h] BYREF
  _OWORD *v19; // [rsp+40h] [rbp-50h] BYREF
  __int64 v20; // [rsp+48h] [rbp-48h]
  _OWORD v21[4]; // [rsp+50h] [rbp-40h] BYREF

  if ( a2 )
  {
    v17[0] = (__int64)v18;
    sub_26E9140(v17, a2, (__int64)&a2[a3]);
  }
  else
  {
    v17[1] = 0;
    v17[0] = (__int64)v18;
    LOBYTE(v18[0]) = 0;
  }
  v6 = (__m128i *)sub_2241130((unsigned __int64 *)v17, 0, 0, "\n*** Pseudo Probe Verification After ", 0x25u);
  v19 = v21;
  if ( (__m128i *)v6->m128i_i64[0] == &v6[1] )
  {
    v21[0] = _mm_loadu_si128(v6 + 1);
  }
  else
  {
    v19 = (_OWORD *)v6->m128i_i64[0];
    *(_QWORD *)&v21[0] = v6[1].m128i_i64[0];
  }
  v7 = v6->m128i_i64[1];
  v6[1].m128i_i8[0] = 0;
  v20 = v7;
  v6->m128i_i64[0] = (__int64)v6[1].m128i_i64;
  v6->m128i_i64[1] = 0;
  if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v20) <= 4 )
    sub_4262D8((__int64)"basic_string::append");
  v8 = " ***\n";
  v9 = (__m128i *)sub_2241490((unsigned __int64 *)&v19, " ***\n", 5u);
  v14 = &v16;
  if ( (__m128i *)v9->m128i_i64[0] == &v9[1] )
  {
    v16 = _mm_loadu_si128(v9 + 1);
  }
  else
  {
    v14 = (__m128i *)v9->m128i_i64[0];
    v16.m128i_i64[0] = v9[1].m128i_i64[0];
  }
  v10 = v9->m128i_u64[1];
  v9[1].m128i_i8[0] = 0;
  v15 = v10;
  v9->m128i_i64[0] = (__int64)v9[1].m128i_i64;
  v11 = v19;
  v9->m128i_i64[1] = 0;
  if ( v11 != v21 )
  {
    v8 = (char *)(*(_QWORD *)&v21[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)v11);
  }
  v12 = v17[0];
  if ( (_QWORD *)v17[0] != v18 )
  {
    v8 = (char *)(v18[0] + 1LL);
    j_j___libc_free_0(v17[0]);
  }
  v13 = sub_C5F790(v12, (__int64)v8);
  sub_CB6200(v13, (unsigned __int8 *)v14, v15);
  if ( !*a4 )
    goto LABEL_31;
  if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a4 + 24LL))(*a4) == &unk_4C5D162 )
  {
    sub_26ECEB0(a1, *(_QWORD *)(*a4 + 8LL));
    goto LABEL_15;
  }
  if ( !*a4 )
    goto LABEL_31;
  if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a4 + 24LL))(*a4) == &unk_4C5D161 )
  {
    sub_26ECD90(a1, *(_QWORD *)(*a4 + 8LL));
    goto LABEL_15;
  }
  if ( !*a4 )
    goto LABEL_31;
  if ( (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a4 + 24LL))(*a4) == &unk_4C5D118 )
  {
    sub_26ECF00(a1, *(_QWORD *)(*a4 + 8LL));
    goto LABEL_15;
  }
  if ( !*a4 || (_UNKNOWN *)(*(__int64 (__fastcall **)(_QWORD))(*(_QWORD *)*a4 + 24LL))(*a4) != &unk_4C5D160 )
LABEL_31:
    BUG();
  sub_26ECF50(a1, *(_QWORD *)(*a4 + 8LL));
LABEL_15:
  if ( v14 != &v16 )
    j_j___libc_free_0((unsigned __int64)v14);
}
