// Function: sub_2210570
// Address: 0x2210570
//
__int64 __fastcall sub_2210570(
        __int64 a1,
        __int64 a2,
        unsigned int *a3,
        unsigned int *a4,
        unsigned int **a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  unsigned int *v8; // r10
  unsigned int *v9; // r10
  unsigned int *v10; // r11
  __int64 result; // rax
  __int64 v12; // r9
  __int64 v13; // r9
  _QWORD v14[2]; // [rsp+0h] [rbp-10h] BYREF

  v8 = a3;
  v14[0] = a6;
  v14[1] = a7;
  if ( a3 == a4 )
  {
    result = 0;
LABEL_7:
    *a5 = v8;
    *a8 = a6;
  }
  else
  {
    do
    {
      if ( *v8 > 0x10FFFF )
      {
        a6 = v14[0];
        result = 2;
        goto LABEL_7;
      }
      if ( !(unsigned __int8)sub_220FDA0((__int64)v14, *v8) )
      {
        v12 = v14[0];
        *a5 = v9;
        *a8 = v12;
        return 1;
      }
      v8 = v9 + 1;
    }
    while ( v10 != v8 );
    v13 = v14[0];
    *a5 = v8;
    *a8 = v13;
    return 0;
  }
  return result;
}
