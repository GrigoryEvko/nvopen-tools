// Function: sub_1B99570
// Address: 0x1b99570
//
__int64 __fastcall sub_1B99570(__int64 a1, __int64 a2, int a3)
{
  char v3; // r8
  __int64 result; // rax
  __int64 v5; // [rsp+8h] [rbp-28h] BYREF
  __int64 v6; // [rsp+10h] [rbp-20h] BYREF
  int v7; // [rsp+18h] [rbp-18h]

  v6 = a2;
  v7 = a3;
  v3 = sub_1B99450(a1 + 264, &v6, &v5);
  result = 0;
  if ( v3 )
  {
    if ( v5 != *(_QWORD *)(a1 + 272) + 24LL * *(unsigned int *)(a1 + 288) )
      return *(unsigned int *)(v5 + 16);
  }
  return result;
}
