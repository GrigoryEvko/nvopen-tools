// Function: sub_2F60C30
// Address: 0x2f60c30
//
__int64 __fastcall sub_2F60C30(_DWORD *a1, _DWORD *a2)
{
  __int64 result; // rax

  result = *a2 < *a1;
  if ( *a2 > *a1 )
    return 0xFFFFFFFFLL;
  return result;
}
