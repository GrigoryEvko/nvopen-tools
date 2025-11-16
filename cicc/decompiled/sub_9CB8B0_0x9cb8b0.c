// Function: sub_9CB8B0
// Address: 0x9cb8b0
//
__int64 __fastcall sub_9CB8B0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  _BYTE *v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  result = sub_BCC840(a2, a3, a4);
  v5 = *(_BYTE **)(a1 + 2016);
  v6 = result;
  if ( v5 == *(_BYTE **)(a1 + 2024) )
  {
    sub_9CABF0(a1 + 2008, v5, &v6);
    return v6;
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = result;
      v5 = *(_BYTE **)(a1 + 2016);
    }
    *(_QWORD *)(a1 + 2016) = v5 + 8;
  }
  return result;
}
