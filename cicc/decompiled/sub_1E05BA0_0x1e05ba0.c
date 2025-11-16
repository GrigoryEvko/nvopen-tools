// Function: sub_1E05BA0
// Address: 0x1e05ba0
//
__int64 __fastcall sub_1E05BA0(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rdi
  __int64 v10; // r14
  unsigned int v11; // r12d
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 result; // rax
  __int64 *v19; // r14
  __int64 *i; // rbx
  __int64 v21; // rdi

  v5 = sub_16E8750(a2, 2 * a3);
  v6 = sub_1263B40(v5, "[");
  v7 = sub_16E7A90(v6, a3);
  v8 = sub_1263B40(v7, "] ");
  v9 = *(_QWORD *)a1;
  v10 = v8;
  if ( v9 )
    sub_1DD64C0(v9, v8);
  else
    sub_1263B40(v8, " <<exit node>>");
  v11 = a3 + 1;
  v12 = sub_1263B40(v10, " {");
  v13 = sub_16E7A90(v12, *(unsigned int *)(a1 + 48));
  v14 = sub_1263B40(v13, ",");
  v15 = sub_16E7A90(v14, *(unsigned int *)(a1 + 52));
  v16 = sub_1263B40(v15, "} [");
  v17 = sub_16E7A90(v16, *(unsigned int *)(a1 + 16));
  result = sub_1263B40(v17, "]\n");
  v19 = *(__int64 **)(a1 + 32);
  for ( i = *(__int64 **)(a1 + 24); v19 != i; result = sub_1E05BA0(v21, a2, v11) )
    v21 = *i++;
  return result;
}
