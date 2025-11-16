// Function: sub_1371CE0
// Address: 0x1371ce0
//
__int64 __fastcall sub_1371CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int16 v4; // bx
  __int16 v5; // r12
  __int16 v6; // dx
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF
  __int64 v8; // [rsp+18h] [rbp-18h]

  v7 = a1;
  v8 = a2;
  if ( a1 )
  {
    if ( *(_QWORD *)a3 )
    {
      v4 = v8;
      v5 = *(_WORD *)(a3 + 8);
      v7 = sub_16CB530();
      LOWORD(v8) = v6;
      sub_1371BB0((__int64)&v7, (__int16)(v4 - v5));
    }
    else
    {
      v7 = -1;
      LOWORD(v8) = 0x3FFF;
    }
  }
  return v7;
}
