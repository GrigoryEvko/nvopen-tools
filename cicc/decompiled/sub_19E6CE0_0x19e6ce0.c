// Function: sub_19E6CE0
// Address: 0x19e6ce0
//
__int64 __fastcall sub_19E6CE0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v5[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = sub_1422850(*(_QWORD *)(a1 + 32), a2);
  if ( v2 )
    return v2;
  v4 = a2;
  if ( !(unsigned __int8)sub_19E6C30(a1 + 1800, &v4, v5) )
    return v2;
  else
    return *(_QWORD *)(v5[0] + 8LL);
}
