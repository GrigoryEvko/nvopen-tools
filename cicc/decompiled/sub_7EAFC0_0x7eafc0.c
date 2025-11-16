// Function: sub_7EAFC0
// Address: 0x7eafc0
//
__int64 __fastcall sub_7EAFC0(__m128i *a1)
{
  __int64 v1; // r14
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // r12
  __m128i *v5; // r13
  __m128i *v6; // rax
  const __m128i *v7; // rbx
  __int64 v8; // rax
  __m128i *v9; // rsi
  int v10; // edi
  __int64 v12; // [rsp+8h] [rbp-68h]
  int v13; // [rsp+18h] [rbp-58h] BYREF
  int v14; // [rsp+1Ch] [rbp-54h] BYREF
  __int64 v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  __int64 v17; // [rsp+30h] [rbp-40h] BYREF
  __int64 v18[7]; // [rsp+38h] [rbp-38h] BYREF

  v1 = (__int64)a1;
  v2 = a1[7].m128i_i64[1];
  v12 = a1->m128i_i64[0];
  v3 = sub_7E1F20(a1[8].m128i_i64[0]);
  v14 = 0;
  v4 = v3;
  if ( (a1[12].m128i_i8[0] & 2) != 0 )
  {
    sub_7E13F0((__int64)a1, &v16, &v17, &v15, v18);
    if ( unk_4D03EB8 )
    {
      a1 = (__m128i *)&v14;
      sub_7296C0(&v14);
    }
    sub_7E1D00(a1, &v16);
    v5 = (__m128i *)sub_724D50(1);
    sub_7E2DB0(v5, v16, unk_4F06895, v4, 0);
    v6 = (__m128i *)sub_724D50(6);
    v7 = v6;
    if ( v15 )
    {
      if ( (*(_BYTE *)(v15 + 193) & 4) != 0 )
      {
        v8 = sub_7E1C10();
        v9 = (__m128i *)v7;
        sub_72BB40(v8, v7);
      }
      else
      {
        v9 = v6;
        sub_72D3B0(v15, (__int64)v6, 1);
      }
      sub_7EAF80(*(_QWORD *)(v15 + 152), v9);
    }
    else
    {
      sub_7E2DB0(v6, v18[0], unk_4F06895, v4, 0);
    }
    sub_712540(v7, qword_4F18A08, 0, 0, &v13, dword_4F07508);
    sub_724A80(v1, 10);
    *(_BYTE *)(v1 + 172) |= 8u;
    v10 = v14;
    *(_QWORD *)(v1 + 176) = v7;
    v7[7].m128i_i64[1] = (__int64)v5;
    *(_QWORD *)(v1 + 184) = v5;
    sub_729730(v10);
  }
  else
  {
    sub_7E1360((__int64)a1, &v16);
    sub_7E2DB0(a1, v16, unk_4D03F80, v4, 1);
  }
  *(_QWORD *)(v1 + 120) = v2;
  *(_QWORD *)v1 = v12;
  return v12;
}
