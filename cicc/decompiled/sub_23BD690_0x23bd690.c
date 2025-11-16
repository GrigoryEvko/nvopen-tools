// Function: sub_23BD690
// Address: 0x23bd690
//
void __fastcall sub_23BD690(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __m128i *v7; // rbx
  __m128i *v8; // r12
  __int64 v9; // [rsp+8h] [rbp-88h] BYREF
  __int64 v10[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v11[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 (__fastcall *v12)(__int64 *, __int64 *, int); // [rsp+30h] [rbp-60h]
  __int64 (__fastcall *v13)(); // [rsp+38h] [rbp-58h]
  __m128i *v14; // [rsp+40h] [rbp-50h] BYREF
  __m128i *v15; // [rsp+48h] [rbp-48h]
  __int64 v16; // [rsp+50h] [rbp-40h]
  _QWORD v17[7]; // [rsp+58h] [rbp-38h] BYREF

  v2 = sub_904010(*(_QWORD *)(a1 + 40), "<button type=\"button\" class=\"collapsible\">0. ");
  v3 = sub_904010(v2, "Initial IR (by function)</button>\n");
  v4 = sub_904010(v3, "<div class=\"content\">\n");
  sub_904010(v4, "  <p>\n");
  v17[2] = 0x5800000000LL;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17[0] = 0;
  v17[1] = 0;
  sub_23B2720(v11, a2);
  sub_23BC790(v11, &v14);
  if ( v11[0] )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v11[0] + 8LL))(v11[0]);
  v10[0] = (__int64)&v14;
  v10[1] = (__int64)&v14;
  v13 = sub_23C6240;
  v12 = sub_23AE6E0;
  v11[0] = a1;
  sub_23B2720(&v9, a2);
  v5 = sub_23B27D0(&v9);
  sub_23BD210(v10, v5 != 0, (__int64)v11);
  if ( v9 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v9 + 8LL))(v9);
  if ( v12 )
    v12(v11, v11, 3);
  v6 = sub_904010(*(_QWORD *)(a1 + 40), "  </p>\n");
  sub_904010(v6, "</div><br/>\n");
  ++*(_DWORD *)(a1 + 36);
  sub_23B5F50((__int64)v17);
  v7 = v15;
  v8 = v14;
  if ( v15 != v14 )
  {
    do
    {
      if ( (__m128i *)v8->m128i_i64[0] != &v8[1] )
        j_j___libc_free_0(v8->m128i_i64[0]);
      v8 += 2;
    }
    while ( v7 != v8 );
    v8 = v14;
  }
  if ( v8 )
    j_j___libc_free_0((unsigned __int64)v8);
}
