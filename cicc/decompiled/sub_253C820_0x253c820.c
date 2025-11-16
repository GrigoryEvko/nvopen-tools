// Function: sub_253C820
// Address: 0x253c820
//
__int64 *__fastcall sub_253C820(__int64 *a1, __int64 a2)
{
  bool v2; // zf
  const char *v3; // rsi

  v2 = *(_BYTE *)(a2 + 97) == 0;
  v3 = "nofree";
  if ( v2 )
    v3 = "may-free";
  sub_253C590(a1, v3);
  return a1;
}
