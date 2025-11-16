// Function: sub_11F32F0
// Address: 0x11f32f0
//
__int64 __fastcall sub_11F32F0(__int64 *a1, _DWORD *a2, unsigned int a3)
{
  __int64 result; // rax
  __int64 v5; // rsi

  result = *a1;
  v5 = (unsigned int)*a2;
  if ( *a1 + a3 == *a1 + a1[1] )
  {
    a1[1] = v5;
    *(_BYTE *)(result + v5) = 0;
  }
  else
  {
    result = sub_2240CE0(a1, v5, a3 - v5);
  }
  --*a2;
  return result;
}
