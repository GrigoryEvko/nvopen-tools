// Function: sub_6E6080
// Address: 0x6e6080
//
__int64 __fastcall sub_6E6080(int a1, int a2, int a3, int a4, int a5, int a6)
{
  unsigned int *v8; // rdx
  int v10; // eax
  __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-48h]
  unsigned int v14; // [rsp+1Ch] [rbp-34h] BYREF

  v8 = &v14;
  v14 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v8 = 0;
  v13 = (__int64)v8;
  v10 = sub_6E6010();
  sub_8769C0(a1, a2, a3, a4, a5, a6, v10, 0, v13);
  result = v14;
  if ( v14 )
    return sub_6E50A0();
  return result;
}
