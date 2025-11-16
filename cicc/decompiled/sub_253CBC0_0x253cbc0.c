// Function: sub_253CBC0
// Address: 0x253cbc0
//
__int64 *__fastcall sub_253CBC0(__int64 *a1, __int64 a2)
{
  bool v2; // zf
  const char *v3; // rsi

  v2 = *(_BYTE *)(a2 + 97) == 0;
  v3 = "willreturn";
  if ( v2 )
    v3 = "may-noreturn";
  sub_253C590(a1, v3);
  return a1;
}
