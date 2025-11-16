// Function: sub_D03F50
// Address: 0xd03f50
//
__int64 __fastcall sub_D03F50(__int64 a1, __int64 a2, __int64 *a3)
{
  unsigned int v5; // eax
  _BYTE *v6; // rsi
  unsigned int v7; // r13d
  int v8; // ebx
  bool v9; // r8
  int v10; // eax
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v12[0] = *(_QWORD *)(a2 + 72);
  v5 = sub_A746F0(v12);
  v6 = *(_BYTE **)(a2 - 32);
  v7 = v5;
  if ( !*v6 )
  {
    v8 = sub_CF5E30(*a3, (__int64)v6);
    if ( sub_B49990(a2) )
      v8 |= 0x55u;
    v9 = sub_B49A80(a2);
    v10 = v8;
    if ( v9 )
    {
      LOBYTE(v10) = v8 | 0xAA;
      v8 = v10;
    }
    v7 &= v8;
  }
  return v7;
}
