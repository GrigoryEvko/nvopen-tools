// Function: sub_21F6E60
// Address: 0x21f6e60
//
__int64 __fastcall sub_21F6E60(_QWORD *a1, __int64 a2, __int64 a3)
{
  _QWORD *v4; // rax
  __int64 v5; // r12
  _QWORD *v6; // rbx
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 v9[7]; // [rsp+8h] [rbp-38h] BYREF

  v4 = sub_21F6AD0(a1, *(_QWORD *)(a2 - 24));
  v5 = v4[3];
  v6 = v4 + 1;
  if ( (_QWORD *)v5 == v4 + 1 )
    return 1;
  while ( 1 )
  {
    v7 = *(_QWORD *)(v5 + 32);
    v9[0] = 0;
    result = sub_21F29B0((__int64)a1, v7, v9, a3);
    if ( !(_BYTE)result )
      break;
    v5 = sub_220EF30(v5);
    if ( v6 == (_QWORD *)v5 )
      return 1;
  }
  return result;
}
