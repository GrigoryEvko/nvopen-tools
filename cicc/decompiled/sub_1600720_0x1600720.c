// Function: sub_1600720
// Address: 0x1600720
//
__int64 __fastcall sub_1600720(__int64 a1)
{
  __int64 v1; // rbx
  __int64 v2; // r14
  __int16 v3; // r13
  __int64 v4; // r12
  _QWORD *v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rsi
  _BYTE v10[16]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v11; // [rsp+10h] [rbp-40h]

  v1 = *(_QWORD *)(a1 - 48);
  v2 = *(_QWORD *)(a1 - 24);
  v11 = 257;
  v3 = *(_WORD *)(a1 + 18) & 0x7FFF;
  v4 = sub_1648A60(56, 2);
  if ( v4 )
  {
    v5 = *(_QWORD **)v1;
    if ( *(_BYTE *)(*(_QWORD *)v1 + 8LL) == 16 )
    {
      v6 = v5[4];
      v7 = sub_1643320(*v5);
      v8 = sub_16463B0(v7, (unsigned int)v6);
    }
    else
    {
      v8 = sub_1643320(*v5);
    }
    sub_15FEC10(v4, v8, 51, v3, v1, v2, (__int64)v10, 0);
  }
  return v4;
}
