// Function: sub_25612C0
// Address: 0x25612c0
//
void __fastcall sub_25612C0(__int64 *a1, __int64 **a2)
{
  __int64 v2; // r12
  __int64 v4; // rdi
  __int64 v5; // r14
  __int64 v6; // r14
  __int64 v7; // rax
  __int64 v8; // r14
  size_t v9; // rdx
  unsigned __int8 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 *v14; // [rsp+0h] [rbp-70h] BYREF
  __int64 v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int8 *v17; // [rsp+20h] [rbp-50h] BYREF
  size_t v18; // [rsp+28h] [rbp-48h]

  v2 = (__int64)a2;
  sub_253C590((__int64 *)&v14, byte_3F871B3);
  if ( a2[1] )
  {
    v6 = sub_904010(*a1, "digraph \"");
LABEL_9:
    sub_C67200((__int64 *)&v17, (__int64)a2);
    v7 = sub_CB6200(v6, v17, v18);
    sub_904010(v7, "\" {\n");
    sub_2240A30((unsigned __int64 *)&v17);
    v5 = *a1;
    if ( !*(_QWORD *)(v2 + 8) )
      goto LABEL_4;
LABEL_10:
    v8 = sub_904010(v5, "\tlabel=\"");
    sub_C67200((__int64 *)&v17, v2);
    v9 = v18;
    v10 = v17;
    v11 = v8;
    goto LABEL_11;
  }
  v4 = *a1;
  if ( v15 )
  {
    a2 = &v14;
    v6 = sub_904010(v4, "digraph \"");
    goto LABEL_9;
  }
  sub_904010(v4, "digraph unnamed {\n");
  v5 = *a1;
  if ( a2[1] )
    goto LABEL_10;
LABEL_4:
  if ( !v15 )
    goto LABEL_5;
  v13 = sub_904010(v5, "\tlabel=\"");
  sub_C67200((__int64 *)&v17, (__int64)&v14);
  v9 = v18;
  v10 = v17;
  v11 = v13;
LABEL_11:
  v12 = sub_CB6200(v11, v10, v9);
  sub_904010(v12, "\";\n");
  sub_2240A30((unsigned __int64 *)&v17);
  v5 = *a1;
LABEL_5:
  sub_253C590((__int64 *)&v17, byte_3F871B3);
  sub_CB6200(v5, v17, v18);
  sub_2240A30((unsigned __int64 *)&v17);
  sub_904010(*a1, "\n");
  if ( v14 != &v16 )
    j_j___libc_free_0((unsigned __int64)v14);
}
