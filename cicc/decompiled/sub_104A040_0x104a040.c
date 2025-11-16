// Function: sub_104A040
// Address: 0x104a040
//
char __fastcall sub_104A040(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 *v4; // r12

  v4 = *(__int64 **)(a1 + 16);
  if ( v4 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    sub_FDC110(v4);
    j_j___libc_free_0(v4, 8);
    *(_QWORD *)(a1 + 8) = 0;
    return 0;
  }
  if ( !*(_QWORD *)(a1 + 8) )
    return 0;
  return sub_1049B30(a4, (__int64)&unk_4F8D9A8, a2, a3);
}
