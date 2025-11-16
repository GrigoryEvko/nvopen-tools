// Function: sub_396E940
// Address: 0x396e940
//
unsigned __int64 __fastcall sub_396E940(__int64 a1, __int64 *a2)
{
  unsigned __int64 result; // rax
  __int64 v3[5]; // [rsp+8h] [rbp-28h] BYREF

  result = sub_1626D20(*a2);
  if ( result )
  {
    result = sub_399CD50(v3, *(_QWORD *)(a1 + 504), a2, 0);
    if ( v3[0] )
      return sub_161E7C0((__int64)v3, v3[0]);
  }
  return result;
}
