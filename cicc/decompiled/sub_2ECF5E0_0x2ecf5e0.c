// Function: sub_2ECF5E0
// Address: 0x2ecf5e0
//
__int64 __fastcall sub_2ECF5E0(__int64 a1, __int8 *a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __m128i v7; // xmm1
  __m128i v8; // xmm2
  __int8 v9; // dl
  char v10[4]; // [rsp+8h] [rbp-68h] BYREF
  __int64 v11; // [rsp+Ch] [rbp-64h]
  char v12[4]; // [rsp+14h] [rbp-5Ch] BYREF
  __int64 v13; // [rsp+18h] [rbp-58h]
  __m128i v14; // [rsp+20h] [rbp-50h] BYREF
  __m128i v15; // [rsp+30h] [rbp-40h]
  __m128i v16; // [rsp+40h] [rbp-30h]

  v2 = a1 + 864;
  result = sub_2ECEFB0(a1 + 864);
  if ( result )
  {
    *a2 = 0;
  }
  else
  {
    result = sub_2ECEFB0(a1 + 144);
    if ( result )
    {
      *a2 = 1;
    }
    else
    {
      v10[0] = 0;
      v11 = 0;
      sub_2EC90F0(a1, (__int64)v10, 1, v2, a1 + 144);
      v12[0] = 0;
      v13 = 0;
      sub_2EC90F0(a1, (__int64)v12, 1, a1 + 144, v2);
      v4 = *(_QWORD *)(a1 + 1648);
      if ( !v4 || (*(_BYTE *)(v4 + 249) & 4) != 0 || v10[0] != *(_BYTE *)(a1 + 1632) || *(_QWORD *)(a1 + 1636) != v11 )
      {
        *(_BYTE *)(a1 + 1632) = 0;
        *(_WORD *)(a1 + 1668) = 0;
        *(_QWORD *)(a1 + 1636) = 0;
        *(_QWORD *)(a1 + 1648) = 0;
        *(_QWORD *)(a1 + 1656) = 0;
        *(_DWORD *)(a1 + 1664) = 0;
        *(_QWORD *)(a1 + 1672) = 0;
        sub_2ECA220((_QWORD *)a1, v2, (__int64 *)(a1 + 1632));
      }
      v5 = *(_QWORD *)(a1 + 1600);
      if ( !v5 || (*(_BYTE *)(v5 + 249) & 4) != 0 || *(_BYTE *)(a1 + 1584) != v12[0] || *(_QWORD *)(a1 + 1588) != v13 )
      {
        *(_BYTE *)(a1 + 1584) = 0;
        *(_QWORD *)(a1 + 1588) = 0;
        *(_QWORD *)(a1 + 1600) = 0;
        *(_QWORD *)(a1 + 1608) = 0;
        *(_DWORD *)(a1 + 1616) = 0;
        *(_WORD *)(a1 + 1620) = 0;
        *(_QWORD *)(a1 + 1624) = 0;
        sub_2ECA220((_QWORD *)a1, a1 + 144, (__int64 *)(a1 + 1584));
      }
      v6 = *(_QWORD *)a1;
      *(_BYTE *)(a1 + 1608) = 0;
      v7 = _mm_loadu_si128((const __m128i *)(a1 + 1648));
      v8 = _mm_loadu_si128((const __m128i *)(a1 + 1664));
      v14 = _mm_loadu_si128((const __m128i *)(a1 + 1632));
      v15 = v7;
      v16 = v8;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64))(v6 + 144))(a1, &v14, a1 + 1584) )
      {
        result = *(_QWORD *)(a1 + 1600);
        v9 = *(_BYTE *)(a1 + 1609);
      }
      else
      {
        v9 = v15.m128i_i8[9];
        result = v15.m128i_i64[0];
      }
      *a2 = v9;
    }
  }
  return result;
}
