// Function: sub_6BD7B0
// Address: 0x6bd7b0
//
__int64 __fastcall sub_6BD7B0(__int64 a1)
{
  __int64 *v1; // rax
  _QWORD *v2; // rdi

  v1 = *(__int64 **)(a1 + 24);
  do
  {
    v2 = v1;
    v1 = (__int64 *)*v1;
  }
  while ( *((_BYTE *)v1 + 8) != 3 );
  *(_BYTE *)(v2[3] + 40LL) &= ~1u;
  return sub_6BBB10(v2);
}
