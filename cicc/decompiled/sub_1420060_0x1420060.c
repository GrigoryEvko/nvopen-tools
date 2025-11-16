// Function: sub_1420060
// Address: 0x1420060
//
__int64 __fastcall sub_1420060(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax
  _QWORD v4[9]; // [rsp-48h] [rbp-48h] BYREF

  result = 0;
  if ( *(_BYTE *)(a2 + 16) == 54 )
  {
    if ( (*(_QWORD *)(a2 + 48) || *(__int16 *)(a2 + 18) < 0) && sub_1625790(a2, 6) )
    {
      return 1;
    }
    else
    {
      v3 = *(_QWORD *)(a2 - 24);
      v4[1] = -1;
      v4[0] = v3;
      memset(&v4[2], 0, 24);
      return sub_134CBB0(a1, (__int64)v4, 0);
    }
  }
  return result;
}
