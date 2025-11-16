// Function: sub_1E5EFA0
// Address: 0x1e5efa0
//
__int64 __fastcall sub_1E5EFA0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // rbx
  __int64 v5; // r13
  _BYTE *v6; // rax
  __int64 v8; // rax
  __int64 v9; // rax

  sub_1263B40(a2, "=============================--------------------------------\n");
  sub_1263B40(a2, "Inorder PostDominator Tree: ");
  if ( !*(_BYTE *)(a1 + 96) )
  {
    v8 = sub_1263B40(a2, "DFSNumbers invalid: ");
    v9 = sub_16E7A90(v8, *(unsigned int *)(a1 + 100));
    sub_1263B40(v9, " slow queries.");
  }
  sub_1263B40(a2, "\n");
  v3 = *(_QWORD *)(a1 + 80);
  if ( v3 )
    sub_1E05BA0(v3, a2, 1u);
  sub_1263B40(a2, "Roots: ");
  v4 = *(__int64 **)a1;
  v5 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  while ( (__int64 *)v5 != v4 )
  {
    while ( 1 )
    {
      sub_1DD64C0(*v4, a2);
      v6 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) == v6 )
        break;
      ++v4;
      *v6 = 32;
      ++*(_QWORD *)(a2 + 24);
      if ( (__int64 *)v5 == v4 )
        return sub_1263B40(a2, "\n");
    }
    ++v4;
    sub_16E7EE0(a2, " ", 1u);
  }
  return sub_1263B40(a2, "\n");
}
