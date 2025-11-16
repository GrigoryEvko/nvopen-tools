// Function: sub_11DAE50
// Address: 0x11dae50
//
__int64 __fastcall sub_11DAE50(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v4; // r12
  __int64 v6; // [rsp+8h] [rbp-58h]
  _BYTE v7[32]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v8; // [rsp+30h] [rbp-30h]

  v8 = 257;
  BYTE4(v6) = 0;
  if ( a1 )
  {
    BYTE4(v6) = 1;
    LODWORD(v6) = sub_B45210(a1);
  }
  v4 = sub_B33BC0(a2, a3, *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), v6, (__int64)v7);
  sub_BD6B90((unsigned __int8 *)v4, (unsigned __int8 *)a1);
  if ( !v4 || *(_BYTE *)v4 != 85 )
    return v4;
  *(_WORD *)(v4 + 2) = *(_WORD *)(v4 + 2) & 0xFFFC | *(_WORD *)(a1 + 2) & 3;
  return v4;
}
