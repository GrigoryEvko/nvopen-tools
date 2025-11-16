// Function: sub_3088980
// Address: 0x3088980
//
__int64 __fastcall sub_3088980(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 v6; // rbx
  unsigned __int8 *v7; // rsi
  __int64 result; // rax
  unsigned __int8 *v9; // [rsp+8h] [rbp-38h] BYREF

  v4 = sub_30885F0(a1, *(_QWORD *)(a2 - 32));
  v5 = *(_QWORD *)(v4 + 24);
  v6 = v4 + 8;
  if ( v5 == v4 + 8 )
    return 1;
  while ( 1 )
  {
    v7 = *(unsigned __int8 **)(v5 + 32);
    v9 = 0;
    result = sub_30857E0((__int64)a1, v7, &v9, a3);
    if ( !(_BYTE)result )
      break;
    v5 = sub_220EF30(v5);
    if ( v6 == v5 )
      return 1;
  }
  return result;
}
