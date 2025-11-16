// Function: sub_630710
// Address: 0x630710
//
__int64 __fastcall sub_630710(__int64 a1, __int64 a2, int a3)
{
  _BYTE v4[20]; // [rsp+Ch] [rbp-14h] BYREF

  if ( a3 )
    sub_733CF0();
  if ( a2 && *(_BYTE *)(a2 + 16) == 2 && (unsigned int)sub_731890(*(_QWORD *)(a2 + 24), 0, v4)
    || !(unsigned int)sub_733920(a1) )
  {
    sub_732E60(a1, 31, a2);
  }
  return sub_733F40(0);
}
