// Function: sub_32499F0
// Address: 0x32499f0
//
__int64 __fastcall sub_32499F0(__int64 a1, __int64 a2)
{
  unsigned __int64 **v3; // rsi

  v3 = (unsigned __int64 **)(a1 + 120);
  if ( !*(_BYTE *)(a1 + 136) )
    v3 = *(unsigned __int64 ***)(a1 + 112);
  return sub_32499D0(*(__int64 **)(a1 + 16), v3, 65549, a2);
}
