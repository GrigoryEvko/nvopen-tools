// Function: sub_2414CE0
// Address: 0x2414ce0
//
__int64 __fastcall sub_2414CE0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5[5]; // [rsp+8h] [rbp-28h] BYREF

  result = (unsigned __int8)byte_4FE2E28;
  if ( !byte_4FE2E28 )
  {
    result = sub_2207590((__int64)&byte_4FE2E28);
    if ( (_DWORD)result )
    {
      byte_4FE2E30 = (_DWORD)qword_4FE2FA8 != 0;
      result = sub_2207640((__int64)&byte_4FE2E28);
    }
  }
  if ( byte_4FE2E30 )
  {
    v5[0] = a2;
    result = (__int64)sub_FAA780(a1 + 208, v5);
    *(_QWORD *)result = a3;
  }
  return result;
}
