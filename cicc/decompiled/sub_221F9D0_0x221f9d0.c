// Function: sub_221F9D0
// Address: 0x221f9d0
//
__int64 __fastcall sub_221F9D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        long double *a9)
{
  __int64 *v9; // rdi
  __int64 v10; // r8
  int v12; // [rsp+1Ch] [rbp-1Ch] BYREF
  long double v13; // [rsp+20h] [rbp-18h] BYREF

  v9 = *(__int64 **)(a1 + 16);
  v12 = 0;
  v10 = sub_2214360(v9, a2, a3, a4, a5, a6, a7, &v12, (__int64)&v13, 0);
  if ( v12 )
    *a8 = v12;
  else
    *a9 = v13;
  return v10;
}
