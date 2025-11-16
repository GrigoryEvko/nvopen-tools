// Function: sub_396F550
// Address: 0x396f550
//
__int64 __fastcall sub_396F550(__int64 a1)
{
  __int64 v1; // rbp
  __int64 result; // rax
  char *v3; // [rsp-38h] [rbp-38h] BYREF
  char v4; // [rsp-28h] [rbp-28h]
  char v5; // [rsp-27h] [rbp-27h]
  __int64 v6; // [rsp-8h] [rbp-8h]

  result = *(_QWORD *)(a1 + 400);
  if ( !result )
  {
    v6 = v1;
    v5 = 1;
    v3 = "exception";
    v4 = 3;
    result = sub_396F530(a1, (__int64)&v3);
    *(_QWORD *)(a1 + 400) = result;
  }
  return result;
}
