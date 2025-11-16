// Function: sub_134AD70
// Address: 0x134ad70
//
_BYTE *__fastcall sub_134AD70(__int64 a1, __int64 a2)
{
  _BYTE *result; // rax

  sub_1340B80(a1, a2 + 68096);
  sub_133DF90(a1, (__int64 *)(a2 + 80));
  sub_133DF90(a1, (__int64 *)(a2 + 19520));
  sub_133DF90(a1, (__int64 *)(a2 + 38960));
  sub_130B060(a1, (__int64 *)(a2 + 58432));
  sub_130B060(a1, (__int64 *)(a2 + 58672));
  result = sub_130B060(a1, (__int64 *)(a2 + 60456));
  if ( *(_BYTE *)(a2 + 17) )
  {
    sub_130F050(a1, a2 + 62264);
    return sub_1348990(a1, a2 + 62384);
  }
  return result;
}
