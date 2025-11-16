// Function: sub_38BE350
// Address: 0x38be350
//
__int64 __fastcall sub_38BE350(__int64 a1)
{
  __int64 result; // rax
  unsigned __int64 v2; // r12
  __int64 v3; // [rsp-20h] [rbp-20h]

  result = *(_QWORD *)(a1 + 40);
  if ( !result )
  {
    result = sub_22077B0(0x140u);
    if ( result )
    {
      v3 = result;
      sub_390FBD0(result);
      result = v3;
    }
    v2 = *(_QWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 40) = result;
    if ( v2 )
    {
      sub_390FCC0(v2);
      j_j___libc_free_0(v2);
      return *(_QWORD *)(a1 + 40);
    }
  }
  return result;
}
