// Function: sub_38BAC40
// Address: 0x38bac40
//
void __fastcall sub_38BAC40(_QWORD *a1)
{
  __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned __int64 v4; // rdi

  v2 = a1[47];
  v3 = a1[46];
  *a1 = &unk_4A3DFA8;
  if ( v2 != v3 )
  {
    do
    {
      v4 = *(_QWORD *)(v3 + 24);
      if ( v4 )
        j_j___libc_free_0(v4);
      v3 += 48LL;
    }
    while ( v2 != v3 );
    v3 = a1[46];
  }
  if ( v3 )
    j_j___libc_free_0(v3);
}
