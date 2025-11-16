// Function: sub_2FE4DB0
// Address: 0x2fe4db0
//
__int64 __fastcall sub_2FE4DB0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int); // rax
  int v5; // edx
  __int64 result; // rax

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)a1 + 32LL);
  if ( v4 != sub_2D42F30 )
    return ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, __int64))v4)(a1, a2, 0, a4, a2);
  v5 = sub_AE2980(a2, 0)[1];
  result = 2;
  if ( v5 != 1 )
  {
    result = 3;
    if ( v5 != 2 )
    {
      result = 4;
      if ( v5 != 4 )
      {
        result = 5;
        if ( v5 != 8 )
        {
          result = 6;
          if ( v5 != 16 )
          {
            result = 7;
            if ( v5 != 32 )
            {
              result = 8;
              if ( v5 != 64 )
                return 9 * (unsigned int)(v5 == 128);
            }
          }
        }
      }
    }
  }
  return result;
}
