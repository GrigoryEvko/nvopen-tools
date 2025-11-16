// Function: sub_3400D50
// Address: 0x3400d50
//
unsigned __int8 *__fastcall sub_3400D50(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __m128i a5)
{
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int); // r15
  __int64 v8; // rax
  int v9; // edx
  unsigned __int16 v10; // ax
  __int64 v12; // [rsp+8h] [rbp-38h]

  v12 = *(_QWORD *)(a1 + 16);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v12 + 32LL);
  v8 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v7 == sub_2D42F30 )
  {
    v9 = sub_AE2980(v8, 0)[1];
    v10 = 2;
    if ( v9 != 1 )
    {
      v10 = 3;
      if ( v9 != 2 )
      {
        v10 = 4;
        if ( v9 != 4 )
        {
          v10 = 5;
          if ( v9 != 8 )
          {
            v10 = 6;
            if ( v9 != 16 )
            {
              v10 = 7;
              if ( v9 != 32 )
              {
                v10 = 8;
                if ( v9 != 64 )
                  v10 = 9 * (v9 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v10 = v7(v12, v8, 0);
  }
  return sub_3400BD0(a1, a2, a3, v10, 0, a4, a5, 0);
}
