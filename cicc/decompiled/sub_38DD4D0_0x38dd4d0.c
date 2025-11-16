// Function: sub_38DD4D0
// Address: 0x38dd4d0
//
_QWORD *__fastcall sub_38DD4D0(__int64 a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  __int64 v3; // rdi
  const char *v4; // [rsp+0h] [rbp-30h] BYREF
  char v5; // [rsp+10h] [rbp-20h]
  char v6; // [rsp+11h] [rbp-1Fh]

  result = (_QWORD *)sub_38DD280(a1, a2);
  if ( result )
  {
    if ( result[8] )
    {
      v3 = *(_QWORD *)(a1 + 8);
      v6 = 1;
      v5 = 3;
      v4 = "Chained unwind areas can't have handlers!";
      return sub_38BE3D0(v3, a2, (__int64)&v4);
    }
  }
  return result;
}
