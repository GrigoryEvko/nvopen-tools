// Function: sub_169C410
// Address: 0x169c410
//
_QWORD *__fastcall sub_169C410(_QWORD *a1, __int64 *a2, _DWORD *a3, unsigned int a4)
{
  int v6; // eax
  int v7; // edx
  int v8; // r12d
  __int16 *v10[10]; // [rsp+0h] [rbp-50h] BYREF

  v6 = sub_169C2F0((__int64)a2);
  *a3 = v6;
  if ( v6 == 0x80000000 )
  {
    sub_16986C0(v10, a2);
    sub_169C2C0((__int64)v10);
    sub_1698450((__int64)a1, (__int64)v10);
    sub_1698460((__int64)v10);
  }
  else if ( v6 == 0x7FFFFFFF )
  {
    sub_16986C0(a1, a2);
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
    sub_16986C0(v10, a2);
    sub_169C390((__int64)a1, v10, v8, a4);
    sub_1698460((__int64)v10);
  }
  return a1;
}
