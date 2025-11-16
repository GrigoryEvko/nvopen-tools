// Function: sub_ECE000
// Address: 0xece000
//
__int64 __fastcall sub_ECE000(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // rax
  const char *v4; // [rsp+0h] [rbp-40h] BYREF
  char v5; // [rsp+20h] [rbp-20h]
  char v6; // [rsp+21h] [rbp-1Fh]

  if ( *(_DWORD *)sub_ECD7B0(a1) == 9 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    return 0;
  }
  else
  {
    v6 = 1;
    v4 = "expected newline";
    v5 = 3;
    v2 = sub_ECD7B0(a1);
    v3 = sub_ECD6A0(v2);
    return sub_ECDA70(a1, v3, (__int64)&v4, 0, 0);
  }
}
