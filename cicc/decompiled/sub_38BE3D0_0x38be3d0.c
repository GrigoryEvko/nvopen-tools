// Function: sub_38BE3D0
// Address: 0x38be3d0
//
_QWORD *__fastcall sub_38BE3D0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 *v3; // r10
  __int64 *v5; // rdi

  v3 = *(__int64 **)a1;
  *(_BYTE *)(a1 + 1481) = 1;
  if ( v3 )
    return sub_16D14E0(v3, a2, 0, a3, 0, 0, 0, 0, 1u);
  v5 = *(__int64 **)(a1 + 8);
  if ( !v5 )
    sub_16BCFB0(a3, 0);
  return sub_16D14E0(v5, a2, 0, a3, 0, 0, 0, 0, 1u);
}
