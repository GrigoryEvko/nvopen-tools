// Function: sub_11DD710
// Address: 0x11dd710
//
__int64 __fastcall sub_11DD710(__int64 a1, __int64 a2, __int64 a3)
{
  int v3; // edx
  __int64 v4; // rdx
  __int64 v5; // r13
  __int64 v6; // r14
  _QWORD *v7; // r8
  __int64 v9; // r14
  unsigned __int64 v10; // rax
  _BYTE *v11; // rax
  unsigned __int64 v12; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+18h] [rbp-88h]
  unsigned __int64 v15; // [rsp+18h] [rbp-88h]
  __m128i v16; // [rsp+20h] [rbp-80h] BYREF
  __int64 v17; // [rsp+30h] [rbp-70h]
  __int64 v18; // [rsp+38h] [rbp-68h]
  __int64 v19; // [rsp+40h] [rbp-60h]
  __int64 v20; // [rsp+48h] [rbp-58h]
  __int64 v21; // [rsp+50h] [rbp-50h]
  __int64 v22; // [rsp+58h] [rbp-48h]
  __int16 v23; // [rsp+60h] [rbp-40h]

  v3 = *(_DWORD *)(a2 + 4);
  v16.m128i_i32[0] = 0;
  v4 = v3 & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  v14 = *(_QWORD *)(a2 + 32 * (1 - v4));
  v6 = *(_QWORD *)(a2 + 32 * (2 - v4));
  sub_11DA4B0(a2, v16.m128i_i32, 1);
  v16 = (__m128i)*(unsigned __int64 *)(a1 + 16);
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  v23 = 257;
  if ( (unsigned __int8)sub_9B6260(v6, &v16, 0) )
  {
    v16.m128i_i32[0] = 1;
    sub_11DA4B0(a2, v16.m128i_i32, 1);
  }
  if ( *(_BYTE *)v6 != 17 )
    return 0;
  v7 = *(_QWORD **)(v6 + 24);
  if ( *(_DWORD *)(v6 + 32) > 0x40u )
    v7 = (_QWORD *)*v7;
  v12 = (unsigned __int64)v7;
  if ( v7 )
  {
    v9 = v14;
    v10 = sub_98B430(v14, 8u);
    if ( v10 )
    {
      v15 = v10;
      v16.m128i_i32[0] = 1;
      sub_11DA2E0(a2, (unsigned int *)&v16, 1, v10);
      if ( v15 == 1 )
        return v5;
      if ( v15 - 1 <= v12 )
      {
        v11 = sub_11DD500(a1, v9, v5, v15 - 1, a3);
        v5 = (__int64)v11;
        if ( v11 )
        {
          if ( *v11 == 85 )
            *((_WORD *)v11 + 1) = *((_WORD *)v11 + 1) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
          return v5;
        }
      }
    }
    return 0;
  }
  return v5;
}
