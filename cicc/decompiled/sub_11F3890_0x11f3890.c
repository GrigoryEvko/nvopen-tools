// Function: sub_11F3890
// Address: 0x11f3890
//
__int64 __fastcall sub_11F3890(_QWORD *a1)
{
  __int64 result; // rax
  __int64 v3; // r12
  __int64 v4; // rdi
  __int64 v5; // [rsp-20h] [rbp-20h]

  result = a1[3];
  if ( !result )
  {
    v3 = *(_QWORD *)(*a1 + 40LL);
    result = sub_22077B0(112);
    if ( result )
    {
      v5 = result;
      sub_A558A0(result, v3, 1);
      result = v5;
    }
    v4 = a1[3];
    a1[3] = result;
    if ( v4 )
    {
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v4 + 8LL))(v4);
      return a1[3];
    }
  }
  return result;
}
