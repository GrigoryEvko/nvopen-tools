// Function: sub_CA3EA0
// Address: 0xca3ea0
//
bool __fastcall sub_CA3EA0(__int64 a1, __int64 a2)
{
  bool result; // al
  bool v3; // [rsp+Fh] [rbp-71h]
  _QWORD v4[2]; // [rsp+10h] [rbp-70h] BYREF
  __int64 v5; // [rsp+20h] [rbp-60h] BYREF
  char v6; // [rsp+68h] [rbp-18h]

  (*(void (__fastcall **)(_QWORD *, __int64, __int64))(*(_QWORD *)a1 + 40LL))(v4, a1, a2);
  result = 0;
  if ( (v6 & 1) == 0 )
  {
    result = sub_CA3E70((__int64)v4);
    if ( (v6 & 1) == 0 && (__int64 *)v4[0] != &v5 )
    {
      v3 = result;
      j_j___libc_free_0(v4[0], v5 + 1);
      return v3;
    }
  }
  return result;
}
