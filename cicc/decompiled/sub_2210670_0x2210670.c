// Function: sub_2210670
// Address: 0x2210670
//
__int64 __fastcall sub_2210670(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        __int64 a6,
        __int64 a7,
        _QWORD *a8)
{
  unsigned int v10; // eax
  __int64 result; // rax
  __int64 v12; // [rsp+0h] [rbp-28h] BYREF
  __int64 v13; // [rsp+8h] [rbp-20h]

  v12 = a3;
  v13 = a4;
  if ( a3 == a4 )
  {
LABEL_9:
    result = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( a7 == a6 )
      {
        a4 = v12;
        result = 1;
        goto LABEL_8;
      }
      v10 = sub_220F920((__int64)&v12, 0x10FFFFu);
      a4 = v12;
      if ( v10 == -2 )
      {
        result = 1;
        goto LABEL_8;
      }
      if ( v10 > 0x10FFFF )
        break;
      a6 += 4;
      *(_DWORD *)(a6 - 4) = v10;
      if ( v13 == a4 )
        goto LABEL_9;
    }
    result = 2;
  }
LABEL_8:
  *a5 = a4;
  *a8 = a6;
  return result;
}
