// Function: sub_15A7FD0
// Address: 0x15a7fd0
//
__int64 __fastcall sub_15A7FD0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // [rsp+8h] [rbp-8h] BYREF

  if ( (unsigned __int8)sub_16D2B80(a1, a2, 10, &v3) || (result = v3, v3 != (unsigned int)v3) )
    sub_16BD130("not a number, or does not fit in an unsigned int", 1);
  return result;
}
