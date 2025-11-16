// Function: sub_30F9EB0
// Address: 0x30f9eb0
//
char __fastcall sub_30F9EB0(__int64 a1, __int64 a2)
{
  char result; // al

  result = sub_30F9620(a1, a2);
  if ( !result )
    return **(_QWORD **)(*(_QWORD *)(a1 + 56) + 32LL) == *(_QWORD *)(a2 + 40);
  return result;
}
