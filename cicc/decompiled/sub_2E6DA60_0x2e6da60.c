// Function: sub_2E6DA60
// Address: 0x2e6da60
//
__int64 __fastcall sub_2E6DA60(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v8; // rax
  __int64 v9; // rax

  sub_904010(a2, "=============================--------------------------------\n");
  sub_904010(a2, "Inorder Dominator Tree: ");
  if ( !*(_BYTE *)(a1 + 112) )
  {
    v8 = sub_904010(a2, "DFSNumbers invalid: ");
    v9 = sub_CB59D0(v8, *(unsigned int *)(a1 + 116));
    sub_904010(v9, " slow queries.");
  }
  sub_904010(a2, "\n");
  v3 = *(_QWORD *)(a1 + 96);
  if ( v3 )
    sub_2E6D930(v3, a2, 1u);
  sub_904010(a2, "Roots: ");
  v4 = *(__int64 **)a1;
  v5 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  while ( (__int64 *)v5 != v4 )
  {
    v6 = *v4++;
    sub_2E39560(v6, a2);
    sub_904010(a2, " ");
  }
  return sub_904010(a2, "\n");
}
