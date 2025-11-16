// Function: sub_37B5D20
// Address: 0x37b5d20
//
__int64 __fastcall sub_37B5D20(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a1 > *a2;
  if ( *a1 < *a2 )
    return 0xFFFFFFFFLL;
  return result;
}
