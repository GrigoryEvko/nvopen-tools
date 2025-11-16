// Function: sub_267D2C0
// Address: 0x267d2c0
//
__int64 __fastcall sub_267D2C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // r13
  unsigned __int8 *v7; // rax
  __int64 v8; // r13
  unsigned int v9; // r13d

  v2 = *(_QWORD *)(a2 + 24);
  if ( *(_BYTE *)v2 != 85 || a2 != v2 - 32 )
    return 0;
  if ( *(char *)(v2 + 7) < 0 )
  {
    v4 = sub_BD2BC0(*(_QWORD *)(a2 + 24));
    v6 = v4 + v5;
    if ( *(char *)(v2 + 7) < 0 )
      v6 -= sub_BD2BC0(v2);
    if ( (unsigned int)(v6 >> 4) )
      return 0;
  }
  v7 = sub_BD3990(*(unsigned __int8 **)(v2 + 32 * (2LL - (*(_DWORD *)(v2 + 4) & 0x7FFFFFF))), a2);
  v8 = (__int64)v7;
  if ( *v7 )
    return 0;
  if ( !(unsigned __int8)sub_B2DCE0((__int64)v7) )
    return 0;
  v9 = sub_B2D610(v8, 76);
  if ( !(_BYTE)v9 )
    return 0;
  sub_267CD60(*(_QWORD *)a1, v2, "OMP160", 6u);
  sub_B43D60((_QWORD *)v2);
  **(_BYTE **)(a1 + 8) = 1;
  return v9;
}
