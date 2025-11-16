// Function: sub_2215390
// Address: 0x2215390
//
__int64 __fastcall sub_2215390(__int64 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx

  result = *a1;
  if ( *(int *)(*a1 - 8) < 0 )
    *(_DWORD *)(result - 8) = 0;
  v3 = *a2;
  if ( *(int *)(*a2 - 8) < 0 )
    *(_DWORD *)(v3 - 8) = 0;
  *a1 = v3;
  *a2 = result;
  return result;
}
