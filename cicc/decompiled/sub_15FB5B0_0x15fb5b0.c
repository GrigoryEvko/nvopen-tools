// Function: sub_15FB5B0
// Address: 0x15fb5b0
//
__int64 __fastcall sub_15FB5B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v9; // [rsp+8h] [rbp-38h]

  v5 = sub_15A14C0(*a1, a2, a3, a4);
  v9 = *a1;
  v6 = sub_1648A60(56, 2);
  v7 = v6;
  if ( v6 )
    sub_15FB300(v6, 14, v5, (__int64)a1, v9, a2, a3);
  return v7;
}
