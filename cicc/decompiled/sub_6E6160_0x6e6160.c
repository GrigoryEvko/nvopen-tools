// Function: sub_6E6160
// Address: 0x6e6160
//
__int64 __fastcall sub_6E6160(__int64 a1, __int64 a2)
{
  int *v2; // rbx
  unsigned int v3; // eax
  __int64 result; // rax
  unsigned int v5; // [rsp+Ch] [rbp-34h]
  int v6; // [rsp+1Ch] [rbp-24h] BYREF

  v2 = &v6;
  v6 = 0;
  if ( *(char *)(qword_4D03C50 + 18LL) >= 0 )
    v2 = 0;
  v3 = sub_6E6010();
  result = sub_876D90(a1, a1, a2, v3, v2);
  if ( v6 )
  {
    v5 = result;
    sub_6E50A0();
    return v5;
  }
  return result;
}
