// Function: sub_23B2260
// Address: 0x23b2260
//
void __fastcall sub_23B2260(__int64 *a1, unsigned __int64 a2, int a3, __int64 a4)
{
  unsigned __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rax
  _QWORD *v8; // r8
  __int64 v9; // rcx
  int v10; // esi
  __int64 *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  unsigned __int64 *v14; // rax
  size_t v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  _QWORD v25[2]; // [rsp+20h] [rbp-80h] BYREF
  __m128i *v26; // [rsp+30h] [rbp-70h]
  size_t v27; // [rsp+38h] [rbp-68h]
  __m128i v28; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v29; // [rsp+50h] [rbp-50h] BYREF
  __int64 v30; // [rsp+58h] [rbp-48h]
  _QWORD v31[8]; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(_QWORD *)(a4 + 8);
  if ( v4 )
  {
    v29 = *(_QWORD **)(a4 + 8);
    v6 = sub_23AEEF0((_QWORD *)(a2 + 152), (__int64 *)&v29)[2];
    v29 = v31;
    sub_23AEDD0((__int64 *)&v29, *(_BYTE **)(v6 + 16), *(_QWORD *)(v6 + 16) + *(_QWORD *)(v6 + 24));
    v7 = v30;
    v8 = (_QWORD *)(a2 + 152);
    v9 = a4;
    if ( v29 != v31 )
    {
      v23 = v30;
      v21 = v9;
      j_j___libc_free_0((unsigned __int64)v29);
      v8 = (_QWORD *)(a2 + 152);
      v9 = v21;
      v7 = v23;
    }
    v10 = -1;
    if ( v7 )
      v10 = a3;
    v29 = *(_QWORD **)(v9 + 8);
    v11 = (__int64 *)sub_23AEEF0(v8, (__int64 *)&v29)[2];
    v12 = *v11;
    v13 = v11[1];
    v25[0] = v12;
    v25[1] = v13;
    sub_95CA80((__int64 *)&v29, (__int64)v25);
    v14 = sub_2241130((unsigned __int64 *)&v29, 0, 0, "color=", 6u);
    v26 = &v28;
    if ( (unsigned __int64 *)*v14 == v14 + 2 )
    {
      v28 = _mm_loadu_si128((const __m128i *)v14 + 1);
    }
    else
    {
      v26 = (__m128i *)*v14;
      v28.m128i_i64[0] = v14[2];
    }
    v15 = v14[1];
    *((_BYTE *)v14 + 16) = 0;
    v27 = v15;
    *v14 = (unsigned __int64)(v14 + 2);
    v14[1] = 0;
    if ( v29 != v31 )
      j_j___libc_free_0((unsigned __int64)v29);
    v16 = sub_904010(*a1, "\tNode");
    sub_CB5A80(v16, a2);
    if ( v10 != -1 )
    {
      v18 = sub_904010(*a1, ":s");
      sub_CB59F0(v18, v10);
    }
    v17 = sub_904010(*a1, " -> Node");
    sub_CB5A80(v17, v4);
    if ( v27 )
    {
      v19 = sub_904010(*a1, "[");
      v20 = sub_CB6200(v19, (unsigned __int8 *)v26, v27);
      sub_904010(v20, "]");
    }
    sub_904010(*a1, ";\n");
    if ( v26 != &v28 )
      j_j___libc_free_0((unsigned __int64)v26);
  }
}
