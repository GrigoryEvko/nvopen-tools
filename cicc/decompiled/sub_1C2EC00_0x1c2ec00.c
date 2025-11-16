// Function: sub_1C2EC00
// Address: 0x1c2ec00
//
__int64 __fastcall sub_1C2EC00(__int64 a1, __int64 a2)
{
  int v2; // eax
  _DWORD v4[3]; // [rsp+Ch] [rbp-14h] BYREF

  if ( (unsigned __int8)sub_1C2E690(a2, "maxntidx", 8u, v4) )
  {
    v2 = v4[0];
    *(_BYTE *)(a1 + 4) = 1;
    *(_DWORD *)a1 = v2;
  }
  else
  {
    *(_BYTE *)(a1 + 4) = 0;
  }
  return a1;
}
