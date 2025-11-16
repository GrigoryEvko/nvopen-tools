// Function: sub_ED85B0
// Address: 0xed85b0
//
__int64 *__fastcall sub_ED85B0(__int64 *a1, __int64 a2, int a3, void *a4)
{
  *(_DWORD *)(a2 + 8) = a3;
  sub_2240AE0(a2 + 16, a4);
  if ( a3 )
    sub_ED79C0(a1, a3, a4);
  else
    *a1 = 1;
  return a1;
}
