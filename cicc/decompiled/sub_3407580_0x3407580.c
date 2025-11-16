// Function: sub_3407580
// Address: 0x3407580
//
unsigned __int8 *__fastcall sub_3407580(__int64 a1, int a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int); // rbx
  __int64 v10; // rax
  int v11; // edx
  unsigned __int16 v12; // ax
  __int128 v13; // rax
  __int64 v14; // r9
  __int128 v16; // [rsp-30h] [rbp-70h]
  __int64 v17; // [rsp+0h] [rbp-40h]

  v17 = *(_QWORD *)(a1 + 16);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int))(*(_QWORD *)v17 + 32LL);
  v10 = sub_2E79000(*(__int64 **)(a1 + 40));
  if ( v9 == sub_2D42F30 )
  {
    v11 = sub_AE2980(v10, 0)[1];
    v12 = 2;
    if ( v11 != 1 )
    {
      v12 = 3;
      if ( v11 != 2 )
      {
        v12 = 4;
        if ( v11 != 4 )
        {
          v12 = 5;
          if ( v11 != 8 )
          {
            v12 = 6;
            if ( v11 != 16 )
            {
              v12 = 7;
              if ( v11 != 32 )
              {
                v12 = 8;
                if ( v11 != 64 )
                  v12 = 9 * (v11 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v12 = v9(v17, v10, 0);
  }
  *(_QWORD *)&v13 = sub_3400BD0(a1, a2, a5, v12, 0, 1u, a6, 1u);
  *((_QWORD *)&v16 + 1) = a4;
  *(_QWORD *)&v16 = a3;
  return sub_3406EB0((_QWORD *)a1, 0x130u, a5, 262, 0, v14, v16, v13);
}
