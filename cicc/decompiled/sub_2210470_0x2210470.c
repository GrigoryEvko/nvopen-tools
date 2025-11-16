// Function: sub_2210470
// Address: 0x2210470
//
__int64 __fastcall sub_2210470(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        _QWORD *a5,
        char *a6,
        char *a7,
        char **a8)
{
  __int64 v8; // r13
  unsigned int v11; // eax
  __int64 result; // rax
  __int64 v13; // [rsp+0h] [rbp-30h] BYREF
  __int64 v14; // [rsp+8h] [rbp-28h]

  v8 = a3;
  v13 = a3;
  v14 = a4;
  if ( a3 == a4 )
  {
LABEL_12:
    result = 0;
  }
  else
  {
    while ( 1 )
    {
      if ( a7 == a6 )
      {
        v8 = v13;
        result = 0;
        goto LABEL_10;
      }
      v11 = sub_220FAE0((__int64)&v13, 0x10FFFFu);
      if ( v11 == -2 )
      {
        v8 = v13;
        result = 1;
        goto LABEL_10;
      }
      if ( v11 > 0x10FFFF )
        break;
      if ( v11 <= 0xFFFF )
      {
        *(_WORD *)a6 = v11;
        v8 = v13;
        a6 += 2;
        if ( v14 == v13 )
          goto LABEL_12;
      }
      else
      {
        if ( (unsigned __int64)(a7 - a6) <= 2 )
        {
          result = 1;
          goto LABEL_10;
        }
        v8 = v13;
        a6 += 4;
        *((_WORD *)a6 - 1) = (v11 & 0x3FF) - 9216;
        *((_WORD *)a6 - 2) = (v11 >> 10) - 10304;
        if ( v14 == v8 )
          goto LABEL_12;
      }
    }
    v8 = v13;
    result = 2;
  }
LABEL_10:
  *a5 = v8;
  *a8 = a6;
  return result;
}
