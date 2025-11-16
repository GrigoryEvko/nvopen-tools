// Function: sub_85E870
// Address: 0x85e870
//
void __fastcall sub_85E870(__int64 a1, __int64 a2)
{
  __int64 **i; // rbx
  __int64 *v3; // rdi

  if ( *(_BYTE *)(a1 + 140) == 7 )
  {
    for ( i = **(__int64 ****)(a1 + 168); i; i = (__int64 **)*i )
    {
      v3 = i[7];
      if ( v3 )
        sub_85E7F0(v3, a2, ((_BYTE)i[4] & 8) != 0);
    }
  }
}
