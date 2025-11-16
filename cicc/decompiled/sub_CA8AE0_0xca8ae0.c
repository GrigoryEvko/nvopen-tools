// Function: sub_CA8AE0
// Address: 0xca8ae0
//
char *__fastcall sub_CA8AE0(char *a1, unsigned __int64 a2, _QWORD *a3)
{
  unsigned __int64 v3; // rax
  char *v4; // r12
  _BYTE v6[16]; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v7)(_BYTE *, __int64, int); // [rsp+10h] [rbp-30h]
  __int64 (__fastcall *v8)(__int64, __int64 *, _QWORD *, __int64, __int64, __int64); // [rsp+18h] [rbp-28h]

  if ( a2 )
  {
    v3 = a2 - 2;
    if ( v3 <= --a2 )
      a2 = v3;
    ++a1;
  }
  v8 = sub_CA6770;
  v7 = (__int64 (__fastcall *)(_BYTE *, __int64, int))sub_CA61B0;
  v4 = sub_CA67E0(a1, a2, a3, "'\r\n", 3, (__int64)v6);
  if ( v7 )
    v7(v6, (__int64)v6, 3);
  return v4;
}
