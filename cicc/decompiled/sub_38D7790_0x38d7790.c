// Function: sub_38D7790
// Address: 0x38d7790
//
__int64 __fastcall sub_38D7790(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  const char *v3; // [rsp+0h] [rbp-30h] BYREF
  char v4; // [rsp+10h] [rbp-20h]
  char v5; // [rsp+11h] [rbp-1Fh]

  result = *(_QWORD *)(a1 + 16);
  if ( !result )
  {
    v5 = 1;
    v3 = "sec_end";
    v4 = 3;
    result = sub_38BF8E0(a2, (__int64)&v3, 1, 1);
    *(_QWORD *)(a1 + 16) = result;
  }
  return result;
}
