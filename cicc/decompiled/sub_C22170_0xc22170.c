// Function: sub_C22170
// Address: 0xc22170
//
__int64 __fastcall sub_C22170(_QWORD *a1)
{
  __int64 v1; // rdx
  __int64 result; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-50h] BYREF
  char v4; // [rsp+10h] [rbp-40h]
  __int64 v5; // [rsp+20h] [rbp-30h] BYREF
  char v6; // [rsp+30h] [rbp-20h]

  sub_C21E40((__int64)v3, a1);
  if ( (v4 & 1) == 0 || (result = LODWORD(v3[0]), v1 = v3[1], !LODWORD(v3[0])) )
  {
    result = (*(__int64 (__fastcall **)(_QWORD *, _QWORD, __int64))(*a1 + 88LL))(a1, v3[0], v1);
    if ( !(_DWORD)result )
    {
      sub_C21E40((__int64)&v5, a1);
      if ( (v6 & 1) == 0 || (result = (unsigned int)v5, !(_DWORD)v5) )
      {
        if ( v5 == 103 )
        {
          sub_C1AFD0();
          return 0;
        }
        else
        {
          sub_C1AFD0();
          return 2;
        }
      }
    }
  }
  return result;
}
