// Function: sub_1626D30
// Address: 0x1626d30
//
__int64 __fastcall sub_1626D30(__int64 a1)
{
  __int64 v1; // rax
  unsigned int v2; // r8d
  __int64 v3; // rax

  v1 = sub_1626D20(a1);
  v2 = 0;
  if ( v1 )
  {
    v3 = *(_QWORD *)(v1 + 8 * (5LL - *(unsigned int *)(v1 + 8)));
    if ( v3 )
      return *(unsigned __int8 *)(v3 + 49);
  }
  return v2;
}
