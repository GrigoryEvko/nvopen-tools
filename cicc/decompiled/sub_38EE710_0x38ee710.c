// Function: sub_38EE710
// Address: 0x38ee710
//
__int64 __fastcall sub_38EE710(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  __int64 result; // rax

  sub_38E2E70(
    a1,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 456) - 8LL) + 16LL),
    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 456) - 8LL) + 8LL));
  sub_38EB180(a1);
  v2 = *(_QWORD *)(a1 + 456);
  v3 = *(_QWORD *)(v2 - 8);
  if ( v3 )
  {
    j_j___libc_free_0(v3);
    v2 = *(_QWORD *)(a1 + 456);
  }
  result = v2 - 8;
  *(_QWORD *)(a1 + 456) = result;
  return result;
}
