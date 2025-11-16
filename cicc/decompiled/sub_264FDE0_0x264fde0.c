// Function: sub_264FDE0
// Address: 0x264fde0
//
__int64 __fastcall sub_264FDE0(__int64 a1, __int64 a2, __int64 a3)
{
  if ( *(_DWORD *)(a2 + 16) >= *(_DWORD *)(a3 + 16) )
    sub_264FCD0(a1, a3, a2);
  else
    sub_264FCD0(a1, a2, a3);
  return a1;
}
