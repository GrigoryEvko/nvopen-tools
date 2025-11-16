// Function: sub_38D6170
// Address: 0x38d6170
//
__int64 __fastcall sub_38D6170(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  __int64 v7; // [rsp+0h] [rbp-30h] BYREF
  char v8; // [rsp+8h] [rbp-28h]

  sub_38D3EB0((__int64)&v7, a2, a3);
  if ( v8 )
    return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 424LL))(a1, v7, a4);
  else
    return sub_38DDD60(a1, a2, a3, a4);
}
