// Function: sub_C24310
// Address: 0xc24310
//
__int64 __fastcall sub_C24310(_QWORD *a1, __int64 a2)
{
  __m128i *v3; // rsi
  __int64 v4; // rax
  __int32 v5; // ecx
  __int64 result; // rax
  __int64 v7; // [rsp+0h] [rbp-90h] BYREF
  char v8; // [rsp+10h] [rbp-80h]
  __int64 v9; // [rsp+20h] [rbp-70h] BYREF
  char v10; // [rsp+30h] [rbp-60h]
  __int64 v11; // [rsp+40h] [rbp-50h] BYREF
  char v12; // [rsp+50h] [rbp-40h]

  sub_C21E40((__int64)&v7, a1);
  if ( (v8 & 1) == 0 || (result = (unsigned int)v7, !(_DWORD)v7) )
  {
    sub_C21E40((__int64)&v9, a1);
    if ( (v10 & 1) == 0 || (result = (unsigned int)v9, !(_DWORD)v9) )
    {
      sub_C21E40((__int64)&v11, a1);
      if ( (v12 & 1) == 0 || (result = (unsigned int)v11, !(_DWORD)v11) )
      {
        v3 = *(__m128i **)(a2 + 8);
        if ( v3 == *(__m128i **)(a2 + 16) )
        {
          sub_C24150((const __m128i **)a2, v3, &v7, &v9, &v11);
        }
        else
        {
          if ( v3 )
          {
            v4 = v11;
            v5 = v7;
            v3->m128i_i64[1] = v9;
            v3->m128i_i32[0] = v5;
            v3[1].m128i_i64[0] = v4;
            v3 = *(__m128i **)(a2 + 8);
          }
          *(_QWORD *)(a2 + 8) = (char *)v3 + 24;
        }
        sub_C1AFD0();
        return 0;
      }
    }
  }
  return result;
}
