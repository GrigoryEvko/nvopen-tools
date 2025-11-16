// Function: sub_3400EE0
// Address: 0x3400ee0
//
unsigned __int8 *__fastcall sub_3400EE0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4, __m128i a5)
{
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int); // rax
  int v12; // edx
  unsigned __int16 v13; // ax
  __int64 (__fastcall *v15)(__int64, __int64, __int64, __int64); // [rsp+8h] [rbp-38h]

  v8 = *(_QWORD *)(a1 + 16);
  v15 = *(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v8 + 72LL);
  v9 = sub_2E79000(*(__int64 **)(a1 + 40));
  v10 = v9;
  if ( v15 == sub_2FE4D20 )
  {
    v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v8 + 32LL);
    if ( v11 == sub_2D42F30 )
    {
      v12 = sub_AE2980(v10, 0)[1];
      v13 = 2;
      if ( v12 != 1 )
      {
        v13 = 3;
        if ( v12 != 2 )
        {
          v13 = 4;
          if ( v12 != 4 )
          {
            v13 = 5;
            if ( v12 != 8 )
            {
              v13 = 6;
              if ( v12 != 16 )
              {
                v13 = 7;
                if ( v12 != 32 )
                {
                  v13 = 8;
                  if ( v12 != 64 )
                    v13 = 9 * (v12 == 128);
                }
              }
            }
          }
        }
      }
    }
    else
    {
      v13 = v11(v8, v10, 0);
    }
  }
  else
  {
    v13 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v8, v9);
  }
  return sub_3400BD0(a1, a2, a3, v13, 0, a4, a5, 0);
}
