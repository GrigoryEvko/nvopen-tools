// Function: sub_B1B090
// Address: 0xb1b090
//
__int64 __fastcall sub_B1B090(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int8 **v4; // rbx
  __int64 v5; // r13
  unsigned __int8 *v6; // rdi
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
    sub_B1AF60(v3, a2, 1u);
  sub_904010(a2, "Roots: ");
  v4 = *(unsigned __int8 ***)a1;
  v5 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  while ( (unsigned __int8 **)v5 != v4 )
  {
    v6 = *v4++;
    sub_A5BF40(v6, a2, 0, 0);
    sub_904010(a2, " ");
  }
  return sub_904010(a2, "\n");
}
