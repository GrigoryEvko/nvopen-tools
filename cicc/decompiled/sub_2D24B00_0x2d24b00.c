// Function: sub_2D24B00
// Address: 0x2d24b00
//
__int64 *__fastcall sub_2D24B00(unsigned __int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 *v3; // r13
  __int64 *i; // rbx
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  unsigned __int8 *v7; // rsi
  __int64 v8; // rsi
  unsigned __int8 *v9; // rsi
  int v10; // [rsp-48h] [rbp-48h]
  __int64 v11; // [rsp-40h] [rbp-40h]
  unsigned __int8 *v12; // [rsp-38h] [rbp-38h] BYREF
  __int64 v13; // [rsp-30h] [rbp-30h]

  if ( a1 != a2 )
  {
    result = (__int64 *)(a2 - 32);
    if ( a1 < a2 - 32 )
    {
      v3 = (__int64 *)(a2 - 16);
      for ( i = (__int64 *)(a1 + 16); ; i += 4 )
      {
        v5 = (unsigned __int8 *)*i;
        v10 = *((_DWORD *)i - 4);
        v6 = *(i - 1);
        v12 = v5;
        v11 = v6;
        if ( v5 )
        {
          sub_B976B0((__int64)i, v5, (__int64)&v12);
          *i = 0;
        }
        v13 = i[1];
        *((_DWORD *)i - 4) = *((_DWORD *)v3 - 4);
        *(i - 1) = *(v3 - 1);
        if ( i != v3 )
        {
          if ( *i )
            sub_B91220((__int64)i, *i);
          v7 = (unsigned __int8 *)*v3;
          *i = *v3;
          if ( v7 )
          {
            sub_B976B0((__int64)v3, v7, (__int64)i);
            *v3 = 0;
          }
        }
        i[1] = v3[1];
        v8 = *v3;
        *((_DWORD *)v3 - 4) = v10;
        *(v3 - 1) = v11;
        if ( v8 )
          sub_B91220((__int64)v3, v8);
        v9 = v12;
        *v3 = (__int64)v12;
        if ( v9 )
          sub_B976B0((__int64)&v12, v9, (__int64)v3);
        v3[1] = v13;
        result = v3 - 4;
        if ( i + 2 >= v3 - 6 )
          break;
        v3 -= 4;
      }
    }
  }
  return result;
}
