// Function: sub_26F9C90
// Address: 0x26f9c90
//
__int64 __fastcall sub_26F9C90(
        _DWORD *a1,
        unsigned __int64 a2,
        __int64 (__fastcall *a3)(__int64, __int64, __int64),
        __int64 a4)
{
  __int64 result; // rax
  unsigned __int8 v7; // [rsp+Fh] [rbp-61h]
  __int64 v8[2]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v9; // [rsp+20h] [rbp-50h] BYREF
  void *v10[4]; // [rsp+30h] [rbp-40h] BYREF
  __int16 v11; // [rsp+50h] [rbp-20h]

  if ( a2 > 7 )
  {
    if ( *(_QWORD *)((char *)a1 + a2 - 8) == 0x6C6175747269762ELL )
      return 0;
  }
  else if ( a2 <= 3 )
  {
    return 0;
  }
  if ( *a1 != 1398037087 )
    return 0;
  v11 = 1283;
  v10[2] = a1 + 1;
  v10[3] = (void *)(a2 - 4);
  v10[0] = "_ZTI";
  sub_CA0F50(v8, v10);
  result = a3(a4, v8[0], v8[1]);
  if ( (__int64 *)v8[0] != &v9 )
  {
    v7 = result;
    j_j___libc_free_0(v8[0]);
    return v7;
  }
  return result;
}
