// Function: sub_120C430
// Address: 0x120c430
//
__int64 __fastcall sub_120C430(__int64 a1, int a2, _DWORD *a3)
{
  __int64 v3; // rbp
  __int64 v4; // rdi
  unsigned __int64 v5; // rsi
  const char *v7; // [rsp-38h] [rbp-38h] BYREF
  char v8; // [rsp-18h] [rbp-18h]
  char v9; // [rsp-17h] [rbp-17h]
  __int64 v10; // [rsp-8h] [rbp-8h]

  if ( a2 == 424 )
  {
    *a3 = 0;
    return 0;
  }
  else if ( a2 == 425 )
  {
    *a3 = 1;
    return 0;
  }
  else
  {
    v10 = v3;
    v4 = a1 + 176;
    v5 = *(_QWORD *)(v4 + 56);
    v7 = "unknown import kind. Expect definition or declaration.";
    v9 = 1;
    v8 = 3;
    sub_11FD800(v4, v5, (__int64)&v7, 1);
    return 1;
  }
}
