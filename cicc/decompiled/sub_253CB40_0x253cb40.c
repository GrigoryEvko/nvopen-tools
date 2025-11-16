// Function: sub_253CB40
// Address: 0x253cb40
//
__int64 *__fastcall sub_253CB40(__int64 *a1, __int64 a2)
{
  bool v2; // zf
  const char *v3; // rsi

  v2 = *(_BYTE *)(a2 + 97) == 0;
  v3 = "mustprogress";
  if ( v2 )
    v3 = "may-not-progress";
  sub_253C590(a1, v3);
  return a1;
}
