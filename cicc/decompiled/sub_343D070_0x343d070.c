// Function: sub_343D070
// Address: 0x343d070
//
__int64 __fastcall sub_343D070(__int64 a1, unsigned __int8 *a2, __m128i a3)
{
  __int64 v3; // r14
  const char **v4; // rbx
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int); // r15
  __int64 v6; // rax
  int v7; // edx
  unsigned __int16 v8; // ax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v12; // [rsp-10h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 864);
  v4 = *(const char ***)(v3 + 16);
  v5 = (__int64 (__fastcall *)(__int64, __int64, unsigned int))*((_QWORD *)*v4 + 4);
  v6 = sub_2E79000(*(__int64 **)(v3 + 40));
  if ( v5 == sub_2D42F30 )
  {
    v7 = sub_AE2980(v6, 0)[1];
    v8 = 2;
    if ( v7 != 1 )
    {
      v8 = 3;
      if ( v7 != 2 )
      {
        v8 = 4;
        if ( v7 != 4 )
        {
          v8 = 5;
          if ( v7 != 8 )
          {
            v8 = 6;
            if ( v7 != 16 )
            {
              v8 = 7;
              if ( v7 != 32 )
              {
                v8 = 8;
                if ( v7 != 64 )
                  v8 = 9 * (v7 == 128);
              }
            }
          }
        }
      }
    }
  }
  else
  {
    v8 = v5((__int64)v4, v6, 0);
  }
  v9 = sub_33EED90(v3, v4[66386], v8, 0);
  sub_343C9F0(a1, a2, v9, v10, 0, 1, a3, 1);
  return v12;
}
