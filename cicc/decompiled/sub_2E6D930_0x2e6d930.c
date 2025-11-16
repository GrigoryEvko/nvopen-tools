// Function: sub_2E6D930
// Address: 0x2e6d930
//
__int64 __fastcall sub_2E6D930(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  unsigned int v11; // r15d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 *v18; // rbx
  __int64 result; // rax
  __int64 *i; // r14
  __int64 v21; // rdi

  v5 = sub_CB69B0(a2, 2 * a3);
  v6 = sub_904010(v5, "[");
  v7 = sub_CB59D0(v6, a3);
  v8 = sub_904010(v7, "] ");
  v9 = *(_QWORD *)a1;
  v10 = v8;
  if ( v9 )
    sub_2E39560(v9, v8);
  else
    sub_904010(v8, " <<exit node>>");
  v11 = a3 + 1;
  v12 = sub_904010(v10, " {");
  v13 = sub_CB59D0(v12, *(unsigned int *)(a1 + 72));
  v14 = sub_904010(v13, ",");
  v15 = sub_CB59D0(v14, *(unsigned int *)(a1 + 76));
  v16 = sub_904010(v15, "} [");
  v17 = sub_CB59D0(v16, *(unsigned int *)(a1 + 16));
  sub_904010(v17, "]\n");
  v18 = *(__int64 **)(a1 + 24);
  result = *(unsigned int *)(a1 + 32);
  for ( i = &v18[result]; i != v18; result = sub_2E6D930(v21, a2, v11) )
    v21 = *v18++;
  return result;
}
