// Function: sub_2EB4190
// Address: 0x2eb4190
//
__int64 __fastcall sub_2EB4190(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rbx
  __int64 v5; // r13
  __int64 v6; // rdi
  __int64 v8; // rax
  __int64 v9; // rax

  sub_904010(a2, "=============================--------------------------------\n");
  sub_904010(a2, "Inorder PostDominator Tree: ");
  if ( !*(_BYTE *)(a1 + 136) )
  {
    v8 = sub_904010(a2, "DFSNumbers invalid: ");
    v9 = sub_CB59D0(v8, *(unsigned int *)(a1 + 140));
    sub_904010(v9, " slow queries.");
  }
  sub_904010(a2, "\n");
  v3 = *(_QWORD *)(a1 + 120);
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
