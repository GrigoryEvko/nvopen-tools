// Function: sub_AF4AF0
// Address: 0xaf4af0
//
__int64 __fastcall sub_AF4AF0(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  _QWORD *v4; // [rsp+0h] [rbp-30h] BYREF
  __int64 v5; // [rsp+8h] [rbp-28h]
  unsigned __int8 v6; // [rsp+10h] [rbp-20h]

  sub_AF4640((__int64)&v4, a1);
  result = v6;
  if ( v6 )
  {
    if ( !v5 )
    {
      *a2 = 0;
      return result;
    }
    if ( v5 == 2 )
    {
      if ( *v4 == 35 )
        goto LABEL_13;
    }
    else if ( v5 == 3 && *v4 == 16 )
    {
      v3 = v4[2];
      if ( v3 != 34 )
      {
        if ( v3 == 28 )
        {
          *a2 = -v4[1];
          return result;
        }
        return 0;
      }
LABEL_13:
      *a2 = v4[1];
      return result;
    }
    return 0;
  }
  return result;
}
