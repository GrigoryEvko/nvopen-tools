// Function: sub_E99600
// Address: 0xe99600
//
__int64 __fastcall sub_E99600(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 (*v6)(); // rdx
  const char *v7; // [rsp+0h] [rbp-50h] BYREF
  char v8; // [rsp+20h] [rbp-30h]
  char v9; // [rsp+21h] [rbp-2Fh]

  result = sub_E99590((__int64)a1, a2);
  if ( result )
  {
    v4 = result;
    if ( *(_QWORD *)(result + 80) )
    {
      v5 = a1[1];
      v9 = 1;
      v8 = 3;
      v7 = "Not all chained regions terminated!";
      sub_E66880(v5, a2, (__int64)&v7);
    }
    v6 = *(__int64 (**)())(*a1 + 88LL);
    result = 1;
    if ( v6 != sub_E97650 )
      result = ((__int64 (__fastcall *)(_QWORD *))v6)(a1);
    *(_QWORD *)(v4 + 16) = result;
  }
  return result;
}
