// Function: sub_2ECF110
// Address: 0x2ecf110
//
__int64 __fastcall sub_2ECF110(__int64 a1, __int8 *a2)
{
  __int64 v2; // r13
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __m128i v8; // xmm0
  __int64 v9; // rax
  __m128i v10; // xmm1
  __m128i v11; // xmm2
  __int8 v12; // dl
  _BYTE v13[4]; // [rsp+18h] [rbp-78h] BYREF
  __int64 v14; // [rsp+1Ch] [rbp-74h]
  _BYTE v15[4]; // [rsp+24h] [rbp-6Ch] BYREF
  __int64 v16; // [rsp+28h] [rbp-68h]
  __m128i v17; // [rsp+30h] [rbp-60h] BYREF
  __m128i v18; // [rsp+40h] [rbp-50h]
  __m128i v19; // [rsp+50h] [rbp-40h]

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
      v13[0] = 0;
      v14 = 0;
      sub_2EC90F0(a1, (__int64)v13, 0, v2, a1 + 144);
      v15[0] = 0;
      v16 = 0;
      sub_2EC90F0(a1, (__int64)v15, 0, a1 + 144, v2);
      v4 = *(_QWORD *)(a1 + 1648);
      if ( !v4 || (*(_BYTE *)(v4 + 249) & 4) != 0 || v13[0] != *(_BYTE *)(a1 + 1632) || *(_QWORD *)(a1 + 1636) != v14 )
      {
        v5 = *(_QWORD *)(a1 + 136);
        *(_WORD *)(a1 + 1668) = 0;
        *(_BYTE *)(a1 + 1632) = 0;
        *(_QWORD *)(a1 + 1636) = 0;
        *(_QWORD *)(a1 + 1648) = 0;
        *(_QWORD *)(a1 + 1656) = 0;
        *(_DWORD *)(a1 + 1664) = 0;
        *(_QWORD *)(a1 + 1672) = 0;
        sub_2EC9FF0((__int64 *)a1, v2, (__int64 *)v13, v5 + 6248, a1 + 1632);
      }
      v6 = *(_QWORD *)(a1 + 1600);
      if ( !v6 || (*(_BYTE *)(v6 + 249) & 4) != 0 || *(_BYTE *)(a1 + 1584) != v15[0] || *(_QWORD *)(a1 + 1588) != v16 )
      {
        *(_BYTE *)(a1 + 1584) = 0;
        *(_WORD *)(a1 + 1620) = 0;
        v7 = *(_QWORD *)(a1 + 136);
        *(_QWORD *)(a1 + 1588) = 0;
        *(_QWORD *)(a1 + 1600) = 0;
        *(_QWORD *)(a1 + 1608) = 0;
        *(_DWORD *)(a1 + 1616) = 0;
        *(_QWORD *)(a1 + 1624) = 0;
        sub_2EC9FF0((__int64 *)a1, a1 + 144, (__int64 *)v15, v7 + 5376, a1 + 1584);
      }
      v8 = _mm_loadu_si128((const __m128i *)(a1 + 1632));
      v9 = *(_QWORD *)a1;
      v10 = _mm_loadu_si128((const __m128i *)(a1 + 1648));
      v11 = _mm_loadu_si128((const __m128i *)(a1 + 1664));
      *(_BYTE *)(a1 + 1608) = 0;
      v17 = v8;
      v18 = v10;
      v19 = v11;
      if ( (*(unsigned __int8 (__fastcall **)(__int64, __m128i *, __int64, _QWORD))(v9 + 144))(a1, &v17, a1 + 1584, 0) )
      {
        result = *(_QWORD *)(a1 + 1600);
        v12 = *(_BYTE *)(a1 + 1609);
      }
      else
      {
        v12 = v18.m128i_i8[9];
        result = v18.m128i_i64[0];
      }
      *a2 = v12;
    }
  }
  return result;
}
