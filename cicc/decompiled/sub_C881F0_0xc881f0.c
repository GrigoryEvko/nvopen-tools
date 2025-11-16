// Function: sub_C881F0
// Address: 0xc881f0
//
__int64 __fastcall sub_C881F0(
        _BYTE *a1,
        size_t a2,
        _QWORD *a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        _QWORD *a7,
        __int64 a8,
        char a9,
        int a10,
        unsigned int a11,
        _QWORD *a12,
        _BYTE *a13,
        __int64 a14)
{
  __int64 result; // rax
  unsigned int v17; // edx
  __int64 v20; // [rsp+10h] [rbp-50h]
  __pid_t v21[18]; // [rsp+18h] [rbp-48h] BYREF

  sub_C86E50(v21);
  if ( (unsigned __int8)sub_C87C00(v21, a1, a2, a3, a4, a9, a7, a8, a5, a6, a11, a12, 0) )
  {
    if ( a13 )
      *a13 = 0;
    if ( a10 )
    {
      BYTE4(v20) = 1;
      LODWORD(v20) = a10;
    }
    else
    {
      BYTE4(v20) = 0;
    }
    sub_C87260(v21, v20, a12, a14, 0);
    return v17;
  }
  else
  {
    result = 0xFFFFFFFFLL;
    if ( a13 )
      *a13 = 1;
  }
  return result;
}
