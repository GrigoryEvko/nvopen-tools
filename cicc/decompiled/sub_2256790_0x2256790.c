// Function: sub_2256790
// Address: 0x2256790
//
__int64 __fastcall sub_2256790(__int64 a1, _DWORD *a2)
{
  __int64 result; // rax

  result = __wcsftime_l();
  if ( !result )
    *a2 = 0;
  return result;
}
