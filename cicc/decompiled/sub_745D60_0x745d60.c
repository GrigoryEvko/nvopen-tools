// Function: sub_745D60
// Address: 0x745d60
//
__int64 __fastcall sub_745D60(
        const char *a1,
        unsigned __int64 a2,
        _DWORD *a3,
        __int64 (__fastcall **a4)(const char *, _QWORD))
{
  __int64 result; // rax
  _BYTE v7[96]; // [rsp+0h] [rbp-60h] BYREF

  if ( *a3 )
    (*a4)(" ", a4);
  (*a4)("__attribute__((", a4);
  (*a4)(a1, a4);
  (*a4)("(", a4);
  if ( a2 > 9 )
  {
    sub_622470(a2, v7);
  }
  else
  {
    v7[1] = 0;
    v7[0] = a2 + 48;
  }
  (*a4)(v7, a4);
  result = (*a4)(")))", a4);
  *a3 = 1;
  return result;
}
