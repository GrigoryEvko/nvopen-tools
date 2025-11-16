// Function: sub_155F6B0
// Address: 0x155f6b0
//
__int64 __fastcall sub_155F6B0(__int64 a1, __int64 *a2)
{
  __int64 v2; // rsi

  v2 = *a2;
  if ( v2 )
  {
    sub_155F620(a1, v2);
  }
  else
  {
    *(_QWORD *)a1 = 0;
    *(_BYTE *)(a1 + 8) = 1;
  }
  return a1;
}
