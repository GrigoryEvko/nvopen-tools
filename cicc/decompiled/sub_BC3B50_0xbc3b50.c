// Function: sub_BC3B50
// Address: 0xbc3b50
//
__int64 __fastcall sub_BC3B50(__int64 a1)
{
  __int64 v1; // r13
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // [rsp-30h] [rbp-30h] BYREF

  if ( *(_BYTE *)(a1 + 416) )
  {
    v1 = *(_QWORD *)(a1 + 408);
    if ( v1 )
    {
      sub_CA0000(a1, *(_QWORD *)(a1 + 408), 1);
      return sub_CA0000(a1 + 112, v1, 1);
    }
    else
    {
      sub_C9DED0(&v4);
      v3 = v4;
      sub_CA0000(a1, v4, 1);
      result = sub_CA0000(a1 + 112, v3, 1);
      if ( v3 )
        return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v3 + 8LL))(v3);
    }
  }
  return result;
}
