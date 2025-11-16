// Function: sub_C3C390
// Address: 0xc3c390
//
_QWORD *__fastcall sub_C3C390(_QWORD *a1, __int64 *a2, _DWORD *a3, char a4)
{
  int v6; // eax
  int v7; // edx
  int v8; // r12d
  _QWORD v10[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = sub_C3BD20((__int64)a2);
  *a3 = v6;
  if ( v6 == 0x80000000 )
  {
    sub_C33EB0(v10, a2);
    sub_C39170((__int64)v10);
    sub_C338E0((__int64)a1, (__int64)v10);
    sub_C338F0((__int64)v10);
  }
  else if ( v6 == 0x7FFFFFFF )
  {
    sub_C33EB0(a1, a2);
  }
  else
  {
    if ( v6 == -2147483647 )
    {
      v8 = 0;
      v7 = 0;
    }
    else
    {
      v7 = v6 + 1;
      v8 = ~v6;
    }
    *a3 = v7;
    sub_C33EB0(v10, a2);
    sub_C3BDC0((__int64)a1, (__int64)v10, v8, a4);
    sub_C338F0((__int64)v10);
  }
  return a1;
}
