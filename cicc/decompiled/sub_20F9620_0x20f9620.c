// Function: sub_20F9620
// Address: 0x20f9620
//
void __fastcall sub_20F9620(_QWORD *a1)
{
  _QWORD *v1; // rax
  char *v2; // rsi
  __int64 v3; // rcx
  unsigned __int64 v4; // rdx
  unsigned __int64 v5; // rax
  char *v6; // rax
  int v7; // [rsp+Ch] [rbp-4h] BYREF

  v1 = (_QWORD *)a1[2];
  v2 = (char *)a1[4];
  v7 = 0;
  v3 = a1[3];
  v4 = 0xF0F0F0F0F0F0F0F1LL * ((__int64)(v1[1] - *v1) >> 4);
  v5 = (__int64)&v2[-v3] >> 2;
  if ( v4 > v5 )
  {
    sub_1CFD340((__int64)(a1 + 3), v2, v4 - v5, &v7);
  }
  else if ( v4 < v5 )
  {
    v6 = (char *)(v3 + 4 * v4);
    if ( v2 != v6 )
      a1[4] = v6;
  }
}
