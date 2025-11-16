// Function: sub_6EB0E0
// Address: 0x6eb0e0
//
__int64 __fastcall sub_6EB0E0(int a1, int a2, __int64 a3)
{
  int *v3; // r15
  int v5; // eax
  __int64 result; // rax
  __int64 v7; // [rsp+8h] [rbp-48h]
  int v8; // [rsp+1Ch] [rbp-34h] BYREF

  v3 = &v8;
  v8 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v3 = 0;
  v5 = sub_6E6010();
  result = sub_87CAB0(a1, a2, a1, 0, (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0, v5, 0, (__int64)v3, a3);
  if ( v8 )
  {
    v7 = result;
    sub_6E50A0();
    return v7;
  }
  return result;
}
