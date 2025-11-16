// Function: sub_34B9AB0
// Address: 0x34b9ab0
//
__int64 __fastcall sub_34B9AB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  if ( a3
    && (*(_DWORD *)(a3 + 4) & 0x7FFFFFF) != 0
    && (unsigned int)**(unsigned __int8 **)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF)) - 12 > 1 )
  {
    return sub_34B9520(a1, a2, a3, a4, a5);
  }
  else
  {
    return 1;
  }
}
