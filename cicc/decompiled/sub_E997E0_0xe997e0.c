// Function: sub_E997E0
// Address: 0xe997e0
//
__int64 __fastcall sub_E997E0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdi
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  result = sub_E99590(a1, a2);
  if ( result )
  {
    if ( *(_QWORD *)(result + 80) )
    {
      v3 = *(_QWORD *)(a1 + 8);
      v6 = 1;
      v5 = 3;
      v4 = "Chained unwind areas can't have handlers!";
      return sub_E66880(v3, a2, (__int64)&v4);
    }
  }
  return result;
}
