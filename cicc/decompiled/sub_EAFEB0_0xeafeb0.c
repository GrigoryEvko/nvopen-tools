// Function: sub_EAFEB0
// Address: 0xeafeb0
//
__int64 __fastcall sub_EAFEB0(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rdi
  __int64 result; // rax

  sub_EA24B0(
    a1,
    *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 376) - 8LL) + 16LL),
    *(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 376) - 8LL) + 8LL));
  sub_EABFE0(a1);
  if ( *(_DWORD *)sub_ECD7B0(a1) == 9 )
    sub_EABFE0(a1);
  v2 = *(_QWORD *)(a1 + 376);
  v3 = *(_QWORD *)(v2 - 8);
  if ( v3 )
  {
    j_j___libc_free_0(v3, 32);
    v2 = *(_QWORD *)(a1 + 376);
  }
  result = v2 - 8;
  *(_QWORD *)(a1 + 376) = result;
  return result;
}
