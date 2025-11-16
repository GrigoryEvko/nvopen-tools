// Function: sub_2B21DD0
// Address: 0x2b21dd0
//
__int64 __fastcall sub_2B21DD0(_DWORD **a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rdi
  __m128i v6; // rax
  unsigned int v7; // r13d
  unsigned int v8; // eax
  unsigned int v9; // edx
  unsigned int v10; // r14d
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // [rsp+0h] [rbp-90h] BYREF
  unsigned int v14; // [rsp+8h] [rbp-88h]
  __m128i v15; // [rsp+10h] [rbp-80h] BYREF
  __int64 v16; // [rsp+20h] [rbp-70h]
  __int64 v17; // [rsp+28h] [rbp-68h]
  __int64 v18; // [rsp+30h] [rbp-60h]
  __int64 v19; // [rsp+38h] [rbp-58h]
  __int64 v20; // [rsp+40h] [rbp-50h]
  __int64 v21; // [rsp+48h] [rbp-48h]
  __int16 v22; // [rsp+50h] [rbp-40h]

  v4 = *(_QWORD *)(a2 + 8);
  v5 = *((_QWORD *)*a1 + 418);
  if ( (unsigned int)*(unsigned __int8 *)(v4 + 8) - 17 <= 1 )
    v4 = **(_QWORD **)(v4 + 16);
  v6.m128i_i64[0] = sub_9208B0(v5, v4);
  v7 = 0;
  v15 = v6;
  v8 = sub_CA1930(&v15);
  v9 = v8;
  v10 = *a1[1];
  if ( v10 < v8 )
  {
    v14 = v8;
    if ( v8 > 0x40 )
    {
      sub_C43690((__int64)&v13, 0, 0);
      v9 = v14;
      if ( v10 == v14 )
      {
LABEL_10:
        v12 = *((_QWORD *)*a1 + 418);
        v16 = 0;
        v15 = (__m128i)v12;
        v17 = 0;
        v18 = 0;
        v19 = 0;
        v20 = 0;
        v21 = 0;
        v22 = 257;
        v7 = sub_9AC230(a2, (__int64)&v13, &v15, 0);
        if ( v14 > 0x40 && v13 )
          j_j___libc_free_0_0(v13);
        return v7;
      }
    }
    else
    {
      v13 = 0;
    }
    if ( v10 > 0x3F || v9 > 0x40 )
      sub_C43C90(&v13, v10, v9);
    else
      v13 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v10 + 64 - (unsigned __int8)v9) << v10;
    goto LABEL_10;
  }
  return v7;
}
