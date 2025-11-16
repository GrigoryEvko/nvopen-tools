// Function: sub_25110D0
// Address: 0x25110d0
//
void __fastcall sub_25110D0(__int64 *a1, __int64 **a2)
{
  __int64 v2; // r12
  __int64 v4; // rdi
  __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  size_t v9; // rdx
  unsigned __int8 *v10; // rsi
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // [rsp+8h] [rbp-78h]
  __int64 *v15; // [rsp+10h] [rbp-70h] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  unsigned __int8 *v18; // [rsp+30h] [rbp-50h] BYREF
  size_t v19; // [rsp+38h] [rbp-48h]
  _QWORD v20[8]; // [rsp+40h] [rbp-40h] BYREF

  v2 = (__int64)a2;
  sub_25072F0((__int64 *)&v15, byte_3F871B3);
  if ( a2[1] )
  {
    v6 = sub_904010(*a1, "digraph \"");
  }
  else
  {
    v4 = *a1;
    if ( !v16 )
    {
      sub_904010(v4, "digraph unnamed {\n");
      goto LABEL_4;
    }
    a2 = &v15;
    v6 = sub_904010(v4, "digraph \"");
  }
  sub_C67200((__int64 *)&v18, (__int64)a2);
  v7 = sub_CB6200(v6, v18, v19);
  sub_904010(v7, "\" {\n");
  if ( v18 != (unsigned __int8 *)v20 )
  {
    j_j___libc_free_0((unsigned __int64)v18);
    v5 = *a1;
    if ( !*(_QWORD *)(v2 + 8) )
      goto LABEL_5;
LABEL_14:
    v8 = sub_904010(v5, "\tlabel=\"");
    sub_C67200((__int64 *)&v18, v2);
    v9 = v19;
    v10 = v18;
    v11 = v8;
    goto LABEL_15;
  }
LABEL_4:
  v5 = *a1;
  if ( *(_QWORD *)(v2 + 8) )
    goto LABEL_14;
LABEL_5:
  if ( !v16 )
    goto LABEL_6;
  v13 = sub_904010(v5, "\tlabel=\"");
  sub_C67200((__int64 *)&v18, (__int64)&v15);
  v9 = v19;
  v10 = v18;
  v11 = v13;
LABEL_15:
  v12 = sub_CB6200(v11, v10, v9);
  sub_904010(v12, "\";\n");
  if ( v18 != (unsigned __int8 *)v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  v5 = *a1;
LABEL_6:
  v14 = v5;
  sub_25072F0((__int64 *)&v18, byte_3F871B3);
  sub_CB6200(v14, v18, v19);
  if ( v18 != (unsigned __int8 *)v20 )
    j_j___libc_free_0((unsigned __int64)v18);
  sub_904010(*a1, "\n");
  if ( v15 != &v17 )
    j_j___libc_free_0((unsigned __int64)v15);
}
