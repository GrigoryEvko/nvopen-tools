// Function: sub_25980A0
// Address: 0x25980a0
//
__int64 __fastcall sub_25980A0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v2; // rax
  __int64 v3; // rax
  unsigned __int8 v5; // [rsp+Fh] [rbp-31h] BYREF
  __int64 v6; // [rsp+10h] [rbp-30h] BYREF
  unsigned __int8 *v7; // [rsp+18h] [rbp-28h]

  v2 = sub_250CBE0((__int64 *)(a1 + 72), a2);
  sub_250D230((unsigned __int64 *)&v6, (unsigned __int64)v2, 4, 0);
  v3 = sub_252A070(a2, v6, (__int64)v7, a1, 0, 0, 1);
  if ( !v3 )
    return sub_2562110(a1);
  v5 = 0;
  v6 = a1;
  v7 = &v5;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, __int64 (__fastcall *)(__int64 *, __int64, __int64, __int64, unsigned int), __int64 *, _QWORD))(*(_QWORD *)v3 + 120LL))(
         v3,
         sub_2562270,
         &v6,
         0) )
  {
    return v5 ^ 1u;
  }
  else
  {
    return sub_2562110(a1);
  }
}
