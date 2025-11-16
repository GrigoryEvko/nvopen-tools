// Function: sub_E99680
// Address: 0xe99680
//
__int64 __fastcall sub_E99680(_QWORD *a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rbx
  __int64 (*v4)(); // rcx
  __int64 v5; // rdx
  __int64 v6; // rdi
  const char *v7; // [rsp+0h] [rbp-50h] BYREF
  char v8; // [rsp+20h] [rbp-30h]
  char v9; // [rsp+21h] [rbp-2Fh]

  result = sub_E99590((__int64)a1, a2);
  if ( result )
  {
    v3 = result;
    result = *(_QWORD *)(result + 80);
    if ( result )
    {
      v4 = *(__int64 (**)())(*a1 + 88LL);
      v5 = 1;
      if ( v4 != sub_E97650 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD *, __int64 (*)(), __int64))v4)(a1, sub_E97650, 1);
        result = *(_QWORD *)(v3 + 80);
      }
      *(_QWORD *)(v3 + 8) = v5;
      a1[13] = result;
    }
    else
    {
      v6 = a1[1];
      v9 = 1;
      v8 = 3;
      v7 = "End of a chained region outside a chained region!";
      return sub_E66880(v6, a2, (__int64)&v7);
    }
  }
  return result;
}
