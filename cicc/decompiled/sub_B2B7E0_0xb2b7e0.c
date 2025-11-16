// Function: sub_B2B7E0
// Address: 0xb2b7e0
//
__int64 __fastcall sub_B2B7E0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  result = sub_AA64B0(a2, 0);
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 40);
    if ( v3 )
    {
      v4 = sub_BD5C70(a2);
      return sub_BD8AE0(v3, v4);
    }
  }
  return result;
}
