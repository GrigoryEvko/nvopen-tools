// Function: sub_B2B790
// Address: 0xb2b790
//
__int64 __fastcall sub_B2B790(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rdi

  result = sub_AA64B0(a2, a1 - 72);
  if ( (*(_BYTE *)(a2 + 7) & 0x10) != 0 )
  {
    v3 = *(_QWORD *)(a1 + 40);
    if ( v3 )
      return sub_BD8920(v3, a2);
  }
  return result;
}
