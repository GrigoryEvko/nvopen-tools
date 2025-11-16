// Function: sub_33C7E90
// Address: 0x33c7e90
//
__int64 __fastcall sub_33C7E90(__int64 a1, __int64 a2, __int64 a3)
{
  bool v3; // zf
  __int64 v5; // [rsp+0h] [rbp-10h] BYREF
  __int64 v6; // [rsp+8h] [rbp-8h] BYREF

  v3 = *(_QWORD *)(a1 + 40) == 0;
  v5 = a2;
  v6 = a3;
  if ( v3 )
    sub_4263D6(a1, a2, a3);
  return (*(__int64 (__fastcall **)(__int64, __int64 *, __int64 *))(a1 + 48))(a1 + 24, &v5, &v6);
}
