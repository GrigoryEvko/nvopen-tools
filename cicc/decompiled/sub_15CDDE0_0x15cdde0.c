// Function: sub_15CDDE0
// Address: 0x15cdde0
//
__int64 __fastcall sub_15CDDE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 *v3; // rdi
  __int64 v4; // rax
  __int64 v5; // rax

  sub_1263B40(a2, "=============================--------------------------------\n");
  sub_1263B40(a2, "Inorder Dominator Tree: ");
  if ( !*(_BYTE *)(a1 + 72) )
  {
    v4 = sub_1263B40(a2, "DFSNumbers invalid: ");
    v5 = sub_16E7A90(v4, *(unsigned int *)(a1 + 76));
    sub_1263B40(v5, " slow queries.");
  }
  result = sub_1263B40(a2, "\n");
  v3 = *(__int64 **)(a1 + 56);
  if ( v3 )
    return sub_1441290(v3, a2, 1u);
  return result;
}
