// Function: sub_E92830
// Address: 0xe92830
//
__int64 __fastcall sub_E92830(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  result = *(_QWORD *)(a1 + 24);
  if ( !result )
  {
    v8 = 1;
    v6 = "sec_end";
    v7 = 3;
    result = sub_E6C380(a2, (__int64 *)&v6, 1, a4, a5);
    *(_QWORD *)(a1 + 24) = result;
  }
  return result;
}
