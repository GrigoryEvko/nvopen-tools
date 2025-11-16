// Function: sub_15AF8F0
// Address: 0x15af8f0
//
__int64 __fastcall sub_15AF8F0(__int64 a1, unsigned int a2)
{
  __int64 v2; // rdi

  v2 = *(_QWORD *)(a1 + 8 * (a2 - (unsigned __int64)*(unsigned int *)(a1 + 8)));
  if ( v2 )
    return sub_161E970(v2);
  else
    return 0;
}
