// Function: sub_20547E0
// Address: 0x20547e0
//
__int64 __fastcall sub_20547E0(__int64 a1, __int64 *a2, int a3, unsigned int a4)
{
  __int64 v4; // rsi
  __int64 result; // rax

  v4 = *a2;
  result = *(unsigned int *)(v4 + 4LL * a4);
  if ( a3 )
    return (unsigned int)(result - *(_DWORD *)(v4 + 4LL * (unsigned int)(a3 - 1)));
  return result;
}
