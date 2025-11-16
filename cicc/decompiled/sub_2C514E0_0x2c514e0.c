// Function: sub_2C514E0
// Address: 0x2c514e0
//
__int64 __fastcall sub_2C514E0(char a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  bool v6; // cc
  __int64 v7; // rdx
  char v8; // cl
  _QWORD *v9; // rdx
  __int64 v10; // r12
  __int64 result; // rax
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned __int64 v16; // r12
  unsigned __int64 v17; // [rsp+0h] [rbp-20h] BYREF
  __int64 v18; // [rsp+8h] [rbp-18h]

  v4 = 1LL << a1;
  if ( *(_BYTE *)a3 == 17 )
  {
    v17 = sub_9208B0(a4, a2);
    v6 = *(_DWORD *)(a3 + 32) <= 0x40u;
    v18 = v7;
    v8 = v7;
    v9 = *(_QWORD **)(a3 + 24);
    if ( !v6 )
      v9 = (_QWORD *)*v9;
    LOBYTE(v18) = v8;
    v17 = (_QWORD)v9 * ((v17 + 7) >> 3);
    v10 = sub_CA1930(&v17) | v4;
    result = 0xFFFFFFFFLL;
    if ( (v10 & -v10) != 0 )
    {
      _BitScanReverse64(&v12, v10 & -v10);
      return 63 - ((unsigned int)v12 ^ 0x3F);
    }
  }
  else
  {
    v13 = sub_9208B0(a4, a2);
    v18 = v14;
    v17 = (unsigned __int64)(v13 + 7) >> 3;
    v15 = v4 | sub_CA1930(&v17);
    result = 0xFFFFFFFFLL;
    if ( (v15 & -v15) != 0 )
    {
      _BitScanReverse64(&v16, v15 & -v15);
      return 63 - ((unsigned int)v16 ^ 0x3F);
    }
  }
  return result;
}
