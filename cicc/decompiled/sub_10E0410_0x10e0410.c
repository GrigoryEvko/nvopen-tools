// Function: sub_10E0410
// Address: 0x10e0410
//
unsigned __int16 __fastcall sub_10E0410(__int64 a1)
{
  char v1; // r12
  unsigned __int16 result; // ax
  __int64 v3; // rax
  _QWORD v4[3]; // [rsp+8h] [rbp-18h] BYREF

  result = sub_A74820((_QWORD *)(a1 + 72));
  if ( !HIBYTE(result) )
  {
    v3 = *(_QWORD *)(a1 - 32);
    if ( !v3 || *(_BYTE *)v3 )
    {
      LOBYTE(result) = v1;
      HIBYTE(result) = 0;
    }
    else if ( *(_QWORD *)(v3 + 24) == *(_QWORD *)(a1 + 80) )
    {
      v4[0] = *(_QWORD *)(v3 + 120);
      return sub_A74820(v4);
    }
    else
    {
      LOBYTE(result) = v1;
      HIBYTE(result) = 0;
    }
  }
  return result;
}
