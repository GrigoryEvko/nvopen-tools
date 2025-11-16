// Function: sub_710080
// Address: 0x710080
//
__int64 __fastcall sub_710080(const __m128i *a1, __int64 a2, int a3, _DWORD *a4, _BYTE *a5)
{
  __int8 v5; // al
  __int64 result; // rax
  int v9; // r9d
  char v11; // [rsp+1Eh] [rbp-62h] BYREF
  char v12; // [rsp+1Fh] [rbp-61h] BYREF
  int v13; // [rsp+20h] [rbp-60h] BYREF
  int v14; // [rsp+24h] [rbp-5Ch] BYREF
  int v15; // [rsp+28h] [rbp-58h] BYREF
  int v16; // [rsp+2Ch] [rbp-54h] BYREF
  __int16 v17[8]; // [rsp+30h] [rbp-50h] BYREF
  _OWORD v18[4]; // [rsp+40h] [rbp-40h] BYREF

  *a4 = 0;
  *a5 = 5;
  v5 = a1[10].m128i_i8[13];
  if ( v5 == 1 )
  {
    sub_724A80(a2, 1);
    *(__m128i *)(a2 + 176) = _mm_loadu_si128(a1 + 11);
    sub_70FEF0(a2, &v11, &v13, &v15);
    sub_621EE0(v17, v15);
    result = sub_6213D0(a2 + 176, (__int64)v17);
    v9 = a3;
    if ( v13 )
    {
      result = sub_6215A0((__int16 *)(a2 + 176), v15);
      v9 = a3;
    }
    if ( v9 )
    {
      result = sub_621060(a2, (__int64)a1);
      if ( (_DWORD)result )
      {
        result = sub_8D2E30(a1[8].m128i_i64[0]);
        if ( !(_DWORD)result )
        {
          sub_70FEF0((__int64)a1, &v12, &v14, &v16);
          result = (unsigned int)v16;
          if ( v15 >= v16 )
            goto LABEL_16;
          v18[0] = _mm_loadu_si128(a1 + 11);
          if ( v14 && (int)sub_6210B0((__int64)a1, 0) < 0 )
          {
            sub_621EE0(v17, v15 - 1);
            sub_621DB0(v17);
            sub_6213B0((__int64)v18, (__int64)v17);
          }
          else
          {
            sub_6213D0((__int64)v18, (__int64)v17);
          }
          result = sub_621000((__int16 *)v18, v14, a1[11].m128i_i16, v14);
          if ( (_DWORD)result )
          {
            *a4 = 69;
            *a5 = 5;
          }
          else
          {
LABEL_16:
            if ( (a1[10].m128i_i8[9] & 1) == 0 )
            {
              *a4 = 68;
              *a5 = 5;
            }
          }
        }
      }
    }
  }
  else
  {
    if ( v5 != 8 )
      sub_721090(a1);
    sub_724A80(a2, 8);
    *(_QWORD *)(a2 + 176) = a1[11].m128i_i64[0];
    result = a1[11].m128i_i64[1];
    *(_QWORD *)(a2 + 184) = result;
  }
  return result;
}
