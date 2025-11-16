// Function: sub_20A14E0
// Address: 0x20a14e0
//
__int64 __fastcall sub_20A14E0(_QWORD *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rdi

  v2 = a1[9258];
  *a1 = &unk_49FEE48;
  while ( v2 )
  {
    sub_20A1310(*(_QWORD *)(v2 + 24));
    v3 = v2;
    v2 = *(_QWORD *)(v2 + 16);
    j_j___libc_free_0(v3, 48);
  }
  return j___libc_free_0(a1[4]);
}
