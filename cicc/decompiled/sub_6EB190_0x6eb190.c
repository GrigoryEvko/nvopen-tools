// Function: sub_6EB190
// Address: 0x6eb190
//
__int64 __fastcall sub_6EB190(int a1, int a2, int a3, int a4, int a5, int a6)
{
  _BOOL4 v7; // ecx
  int *v11; // rdx
  __int64 result; // rax
  __int64 v13; // [rsp+8h] [rbp-38h]
  int v14; // [rsp+1Ch] [rbp-24h] BYREF

  v7 = 0;
  v14 = 0;
  if ( a6 )
    v7 = sub_6E6010() != 0;
  v11 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) < 0 )
    v11 = &v14;
  result = sub_87CF30(a1, a2, a3, a4, a1, a5, a6, (*(_BYTE *)(qword_4D03C50 + 17LL) & 2) != 0, 0, v7, (__int64)v11);
  if ( v14 )
  {
    v13 = result;
    sub_6E50A0();
    return v13;
  }
  return result;
}
