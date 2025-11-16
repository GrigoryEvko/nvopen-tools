// Function: sub_5CAD90
// Address: 0x5cad90
//
__int64 __fastcall sub_5CAD90(__int64 a1, unsigned __int16 a2, _WORD *a3)
{
  __int64 v4; // rdi
  __int16 v7; // ax
  __int64 v8; // rdi
  __int64 v9[3]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(_QWORD *)(a1 + 32);
  if ( !v4 || !(unsigned int)sub_5CACA0(v4, a1, a2, 0xFFFF, v9) )
    return 0;
  v7 = v9[0];
  if ( v9[0] <= 100 )
  {
    v8 = 1220;
    if ( *(_BYTE *)(a1 + 8) != 47 )
      v8 = 1674;
    sub_684B30(v8, a1 + 56);
    v7 = v9[0];
  }
  *a3 = v7;
  return 1;
}
