// Function: sub_23DDA50
// Address: 0x23dda50
//
__int64 __fastcall sub_23DDA50(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 result; // rax
  __int16 v7; // [rsp-28h] [rbp-28h]

  result = a1[34];
  if ( !result )
  {
    v7 = 1283;
    result = sub_29F3C00(*a1, *(_QWORD *)(*a1 + 168LL), *(_QWORD *)(*a1 + 176LL), 0, a5, a6, "___asan_gen_");
    a1[34] = result;
  }
  return result;
}
