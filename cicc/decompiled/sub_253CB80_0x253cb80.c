// Function: sub_253CB80
// Address: 0x253cb80
//
__int64 *__fastcall sub_253CB80(__int64 *a1, __int64 a2)
{
  bool v2; // zf
  const char *v3; // rsi

  v2 = *(_BYTE *)(a2 + 97) == 0;
  v3 = "noreturn";
  if ( v2 )
    v3 = "may-return";
  sub_253C590(a1, v3);
  return a1;
}
