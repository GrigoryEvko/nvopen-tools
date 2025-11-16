// Function: sub_2516660
// Address: 0x2516660
//
__int64 __fastcall sub_2516660(__int64 a1, __m128i *a2, char a3)
{
  unsigned int v5; // r8d
  unsigned __int8 *v7; // rax
  __int64 v8; // r14
  unsigned __int8 v9; // al
  __int64 *v10; // r15
  unsigned int v11; // ebx
  __int64 *v12; // rdi
  __int64 *v13; // rax
  unsigned __int8 v14; // [rsp+17h] [rbp-69h]
  __int64 *v15; // [rsp+18h] [rbp-68h]
  __int64 v16; // [rsp+28h] [rbp-58h] BYREF
  __int64 *v17; // [rsp+30h] [rbp-50h] BYREF
  __int64 v18; // [rsp+38h] [rbp-48h]
  _BYTE v19[64]; // [rsp+40h] [rbp-40h] BYREF

  LODWORD(v17) = 39;
  v5 = sub_2516400(a1, a2, (__int64)&v17, 1, a3, 39);
  if ( !(_BYTE)v5 )
  {
    v7 = sub_250CBE0(a2->m128i_i64, (__int64)a2);
    v5 = 0;
    v8 = (__int64)v7;
    if ( v7 )
    {
      v9 = sub_B2D610((__int64)v7, 6);
      v5 = 0;
      v14 = v9;
      if ( !v9 )
      {
        v17 = (__int64 *)v19;
        v18 = 0x200000000LL;
        LODWORD(v16) = 92;
        sub_2515D00(a1, a2, (int *)&v16, 1, (__int64)&v17, a3);
        v10 = v17;
        v15 = &v17[(unsigned int)v18];
        if ( v17 != v15 )
        {
          v11 = 255;
          do
          {
            v12 = v10++;
            v11 &= sub_A71E40(v12);
          }
          while ( v15 != v10 );
          if ( !(v11 & 2 | (v11 >> 6) & 2 | ((unsigned __int8)(v11 >> 4) | (unsigned __int8)(v11 >> 2)) & 2) )
          {
            v13 = (__int64 *)sub_B2BE50(v8);
            v16 = sub_A778C0(v13, 39, 0);
            sub_2516380(a1, a2->m128i_i64, (__int64)&v16, 1, 0);
            v14 = 1;
          }
          v10 = v17;
        }
        if ( v10 != (__int64 *)v19 )
          _libc_free((unsigned __int64)v10);
        return v14;
      }
    }
  }
  return v5;
}
