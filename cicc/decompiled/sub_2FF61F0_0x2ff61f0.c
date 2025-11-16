// Function: sub_2FF61F0
// Address: 0x2ff61f0
//
__int64 __fastcall sub_2FF61F0(__int64 a1)
{
  unsigned __int64 *v1; // r13
  unsigned __int64 *v2; // r12

  v1 = *(unsigned __int64 **)(a1 + 232);
  v2 = *(unsigned __int64 **)(a1 + 224);
  *(_QWORD *)a1 = &unk_49E3560;
  if ( v1 != v2 )
  {
    do
    {
      if ( *v2 )
        j_j___libc_free_0(*v2);
      v2 += 3;
    }
    while ( v1 != v2 );
    v2 = *(unsigned __int64 **)(a1 + 224);
  }
  if ( v2 )
    j_j___libc_free_0((unsigned __int64)v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 200), 8LL * *(unsigned int *)(a1 + 216), 4);
  return sub_C7D6A0(*(_QWORD *)(a1 + 168), 8LL * *(unsigned int *)(a1 + 184), 4);
}
