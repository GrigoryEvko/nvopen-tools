// Function: sub_38DD370
// Address: 0x38dd370
//
_QWORD *__fastcall sub_38DD370(_QWORD *a1, unsigned __int64 a2)
{
  _QWORD *result; // rax
  _QWORD *v3; // rbx
  __int64 (*v4)(); // rcx
  __int64 v5; // rdx
  __int64 v6; // rdi
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+10h] [rbp-30h]
  char v9; // [rsp+11h] [rbp-2Fh]

  result = (_QWORD *)sub_38DD280((__int64)a1, a2);
  if ( result )
  {
    v3 = result;
    result = (_QWORD *)result[8];
    if ( result )
    {
      v4 = *(__int64 (**)())(*a1 + 16LL);
      v5 = 1;
      if ( v4 != sub_38DBC10 )
      {
        v5 = ((__int64 (__fastcall *)(_QWORD *, __int64 (*)(), __int64))v4)(a1, sub_38DBC10, 1);
        result = (_QWORD *)v3[8];
      }
      v3[1] = v5;
      a1[9] = result;
    }
    else
    {
      v6 = a1[1];
      v9 = 1;
      v8 = 3;
      v7 = "End of a chained region outside a chained region!";
      return sub_38BE3D0(v6, a2, (__int64)&v7);
    }
  }
  return result;
}
