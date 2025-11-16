// Function: sub_20F9690
// Address: 0x20f9690
//
void __fastcall sub_20F9690(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rdx
  char *v3; // rsi
  __int64 v4; // rcx
  unsigned __int64 v5; // rdx
  unsigned __int64 v6; // rax
  char *v7; // rax
  int v8; // [rsp+Ch] [rbp-4h] BYREF

  a1[2] = a2;
  v2 = a2[1] - *a2;
  v3 = (char *)a1[4];
  v8 = 0;
  v4 = a1[3];
  v5 = 0xF0F0F0F0F0F0F0F1LL * (v2 >> 4);
  v6 = (__int64)&v3[-v4] >> 2;
  if ( v5 > v6 )
  {
    sub_1CFD340((__int64)(a1 + 3), v3, v5 - v6, &v8);
  }
  else if ( v5 < v6 )
  {
    v7 = (char *)(v4 + 4 * v5);
    if ( v3 != v7 )
      a1[4] = v7;
  }
}
