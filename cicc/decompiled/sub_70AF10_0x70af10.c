// Function: sub_70AF10
// Address: 0x70af10
//
__int64 __fastcall sub_70AF10(unsigned __int8 a1, __int64 a2, __m128i *a3, int *a4, _DWORD *a5)
{
  __int64 result; // rax
  int v9; // [rsp+14h] [rbp-5Ch] BYREF
  __int64 v10; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v11[80]; // [rsp+20h] [rbp-50h] BYREF

  *a5 = 0;
  v10 = 0;
  v9 = 0;
  sub_70A570(a2, (__int64)v11, &v10, &v9);
  sub_70A7F0((__int64)v11, v10, 0, a1, a3, v9, a4, a5);
  result = a1;
  if ( *a4 )
  {
    if ( !HIDWORD(qword_4F077B4) )
      return result;
    *a4 = 0;
  }
  if ( a1 != 14 && a1 > 8u )
  {
    result = (int)dword_4D04020[a1];
    if ( result <= v10 )
      *a4 = 1;
  }
  return result;
}
