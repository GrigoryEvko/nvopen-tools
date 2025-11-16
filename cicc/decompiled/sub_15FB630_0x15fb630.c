// Function: sub_15FB630
// Address: 0x15fb630
//
__int64 __fastcall sub_15FB630(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // rax
  __int64 v6; // r12
  __int64 v8; // [rsp+8h] [rbp-38h]

  v4 = sub_15A04A0((_QWORD **)*a1);
  v8 = *a1;
  v5 = sub_1648A60(56, 2);
  v6 = v5;
  if ( v5 )
    sub_15FB300(v5, 28, (__int64)a1, v4, v8, a2, a3);
  return v6;
}
