// Function: sub_21CAB10
// Address: 0x21cab10
//
__int64 __fastcall sub_21CAB10(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi

  v2 = a1[9258];
  *a1 = &unk_49FEE48;
  while ( v2 )
  {
    sub_21CA8E0(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  j___libc_free_0(a1[4]);
  return j_j___libc_free_0(a1, 81568);
}
