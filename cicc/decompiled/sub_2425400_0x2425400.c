// Function: sub_2425400
// Address: 0x2425400
//
__int64 __fastcall sub_2425400(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r8
  __int64 v9; // r12
  _QWORD v11[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v12; // [rsp+20h] [rbp-30h]

  v8 = *(_QWORD *)(a1 + 128);
  v11[0] = a3;
  v11[1] = a4;
  v12 = 261;
  v9 = sub_B2CE20(a2, 7, 0, (__int64)v11, v8);
  *(_BYTE *)(v9 + 32) = *(_BYTE *)(v9 + 32) & 0x3F | 0x80;
  sub_B2CD30(v9, 41);
  if ( *(_BYTE *)(a1 + 6) )
    sub_B2CD30(v9, 35);
  sub_2A3ED80(*(_QWORD *)(a1 + 128), v9, a5, a6);
  return v9;
}
