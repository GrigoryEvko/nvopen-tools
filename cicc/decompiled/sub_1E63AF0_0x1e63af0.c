// Function: sub_1E63AF0
// Address: 0x1e63af0
//
__int64 __fastcall sub_1E63AF0(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // r12

  result = sub_1E63940(a1 + 40);
  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 )
  {
    sub_1E63510(*(_QWORD *)(a1 + 32));
    result = j_j___libc_free_0(v2, 112);
  }
  *(_QWORD *)(a1 + 32) = 0;
  return result;
}
