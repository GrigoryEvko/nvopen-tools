// Function: sub_22EC3A0
// Address: 0x22ec3a0
//
__int64 __fastcall sub_22EC3A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  __int64 result; // rax
  __int64 v6; // r15
  __int64 v9; // rcx
  __int64 i; // [rsp+8h] [rbp-38h]

  result = a4 + 24;
  v6 = *(_QWORD *)(a4 + 32);
  for ( i = a4 + 24; i != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v9 = v6 - 56;
    if ( !v6 )
      v9 = 0;
    result = sub_22EC110(a1, a2, a3, v9, a5);
  }
  return result;
}
