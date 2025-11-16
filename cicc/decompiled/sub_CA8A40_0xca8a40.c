// Function: sub_CA8A40
// Address: 0xca8a40
//
char *__fastcall sub_CA8A40(__int64 a1, char *a2, unsigned __int64 a3, _QWORD *a4)
{
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rdx
  char *v8; // r12
  _QWORD v10[2]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v11)(_QWORD *, _QWORD *, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v12)(); // [rsp+18h] [rbp-28h]

  v6 = a3;
  if ( a3 )
  {
    v7 = a3 - 2;
    if ( v7 <= --v6 )
      v6 = v7;
    ++a2;
  }
  v10[0] = a1;
  v12 = sub_CA94A0;
  v11 = sub_CA61C0;
  v8 = sub_CA67E0(a2, v6, a4, "\\\r\n", 3, (__int64)v10);
  if ( v11 )
    v11(v10, v10, 3);
  return v8;
}
