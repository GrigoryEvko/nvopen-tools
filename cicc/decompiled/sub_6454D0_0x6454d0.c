// Function: sub_6454D0
// Address: 0x6454d0
//
__int64 __fastcall sub_6454D0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-28h] BYREF
  __int64 v4; // [rsp+10h] [rbp-20h] BYREF
  __int64 v5; // [rsp+18h] [rbp-18h] BYREF

  result = sub_623FC0(a1, &v3, &v4, &v5);
  if ( (_DWORD)result )
  {
    sub_6854C0(473, a2, v5);
    return 1;
  }
  return result;
}
