// Function: sub_38DD2F0
// Address: 0x38dd2f0
//
__int64 __fastcall sub_38DD2F0(_QWORD *a1, unsigned __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rbx
  __int64 v5; // rdi
  __int64 (*v6)(); // rdx
  const char *v7; // [rsp+0h] [rbp-40h] BYREF
  char v8; // [rsp+10h] [rbp-30h]
  char v9; // [rsp+11h] [rbp-2Fh]

  result = sub_38DD280((__int64)a1, a2);
  if ( result )
  {
    v4 = result;
    if ( *(_QWORD *)(result + 64) )
    {
      v5 = a1[1];
      v9 = 1;
      v8 = 3;
      v7 = "Not all chained regions terminated!";
      sub_38BE3D0(v5, a2, (__int64)&v7);
    }
    v6 = *(__int64 (**)())(*a1 + 16LL);
    result = 1;
    if ( v6 != sub_38DBC10 )
      result = ((__int64 (__fastcall *)(_QWORD *))v6)(a1);
    *(_QWORD *)(v4 + 8) = result;
  }
  return result;
}
