// Function: sub_14F0050
// Address: 0x14f0050
//
__int64 __fastcall sub_14F0050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  result = sub_1644060(a2, a3, a4);
  v5 = *(_BYTE **)(a1 + 1792);
  v6 = result;
  if ( v5 == *(_BYTE **)(a1 + 1800) )
  {
    sub_14EFD20(a1 + 1784, v5, &v6);
    return v6;
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = result;
      v5 = *(_BYTE **)(a1 + 1792);
    }
    *(_QWORD *)(a1 + 1792) = v5 + 8;
  }
  return result;
}
