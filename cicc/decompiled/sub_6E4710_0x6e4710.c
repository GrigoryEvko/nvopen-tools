// Function: sub_6E4710
// Address: 0x6e4710
//
__int64 __fastcall sub_6E4710(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // r12
  __int64 v8; // r12
  void (__fastcall *v9)(__int64, __int64); // [rsp+0h] [rbp-F0h] BYREF
  _QWORD *(__fastcall *v10)(__int64); // [rsp+8h] [rbp-E8h]

  result = *(unsigned __int8 *)(a1 + 16);
  if ( (_BYTE)result == 1 )
  {
    v8 = *(_QWORD *)(a1 + 144);
    if ( v8 )
    {
      sub_76C7C0(&v9, a2, a3, a4, a5, a6);
      v9 = sub_6DEBF0;
      v10 = sub_6E01F0;
      result = sub_76CDC0(v8);
      *(_QWORD *)(a1 + 144) = 0;
    }
  }
  else if ( (_BYTE)result == 2 )
  {
    v7 = *(_QWORD *)(a1 + 288);
    if ( v7 )
    {
      sub_76C7C0(&v9, a2, a3, a4, a5, a6);
      v9 = sub_6DEBF0;
      v10 = sub_6E01F0;
      result = sub_76CDC0(v7);
      *(_QWORD *)(a1 + 288) = 0;
    }
  }
  return result;
}
