// Function: sub_3289F80
// Address: 0x3289f80
//
__int64 __fastcall sub_3289F80(unsigned int *a1, __int64 a2, __int64 a3)
{
  unsigned __int16 v3; // cx
  char v5; // r12
  _QWORD v6[6]; // [rsp+0h] [rbp-30h] BYREF

  v6[0] = a2;
  v6[1] = a3;
  if ( (_WORD)a2 )
  {
    v3 = a2 - 17;
    if ( (unsigned __int16)(a2 - 10) > 6u && (unsigned __int16)(a2 - 126) > 0x31u )
    {
      if ( v3 <= 0xD3u )
        return a1[17];
      return a1[15];
    }
    if ( v3 <= 0xD3u )
      return a1[17];
  }
  else
  {
    v5 = sub_3007030((__int64)v6);
    if ( sub_30070B0((__int64)v6) )
      return a1[17];
    if ( !v5 )
      return a1[15];
  }
  return a1[16];
}
