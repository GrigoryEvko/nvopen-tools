// Function: sub_1315160
// Address: 0x1315160
//
__int64 __fastcall sub_1315160(__int64 a1, __int64 a2, char a3, unsigned __int8 a4)
{
  __int64 result; // rax
  __int64 v7; // r15
  __int64 v8; // [rsp-10h] [rbp-40h]

  if ( a4 )
    sub_130EC10(a1, a2 + 72912);
  result = sub_1314B60(a1, a2, a2 + 69320, *(volatile signed __int64 **)(a2 + 72896), a2 + 10728, a3, a4);
  if ( !(_BYTE)result )
  {
    v7 = sub_13427E0(a2 + 30280);
    if ( sub_13427E0(a2 + 39936) + v7 || (result = sub_130C0D0(a2 + 10672, 2), result > 0) )
    {
      sub_1314B60(a1, a2, a2 + 71104, (volatile signed __int64 *)(*(_QWORD *)(a2 + 72896) + 24LL), a2 + 30168, a3, a4);
      return v8;
    }
  }
  return result;
}
