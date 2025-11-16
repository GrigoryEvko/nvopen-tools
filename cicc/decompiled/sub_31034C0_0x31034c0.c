// Function: sub_31034C0
// Address: 0x31034c0
//
__int64 __fastcall sub_31034C0(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r13

  sub_30EC4F0(a1 + 48);
  sub_30EC4F0(a1 + 88);
  *(_BYTE *)(a1 + 40) = 0;
  v2 = *(__int64 **)(a2 + 32);
  v3 = *(__int64 **)(a2 + 40);
  if ( v2 != v3 )
  {
    while ( !sub_30ED150(a1 + 48, *v2) )
    {
      if ( v3 == ++v2 )
        return sub_3103310(a1, a2);
    }
    *(_BYTE *)(a1 + 40) = 1;
  }
  return sub_3103310(a1, a2);
}
