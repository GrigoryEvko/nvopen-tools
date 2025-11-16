// Function: sub_2A7FEF0
// Address: 0x2a7fef0
//
__int64 __fastcall sub_2A7FEF0(__int64 a1, __int64 a2)
{
  unsigned __int8 *v3; // r12
  __int64 result; // rax
  _BYTE *v5; // rdi
  __int64 v6; // rax
  const char *v7; // [rsp+0h] [rbp-50h] BYREF
  __int16 v8; // [rsp+20h] [rbp-30h]

  v3 = sub_BA8CD0(a2, *(_QWORD *)(a1 + 16), *(_QWORD *)(a1 + 24), 0);
  result = 0;
  if ( v3 )
  {
    sub_2A7FD40(a2, (__int64)v3, a1 + 16, a1 + 48);
    v5 = sub_BA8CD0(a2, *(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 56), 0);
    if ( v5 )
    {
      v6 = sub_BD5C70((__int64)v5);
      sub_BD6500((__int64)v3, v6);
      return 1;
    }
    else
    {
      v7 = (const char *)(a1 + 48);
      v8 = 260;
      sub_BD6B50(v3, &v7);
      return 1;
    }
  }
  return result;
}
