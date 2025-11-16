// Function: sub_3401C20
// Address: 0x3401c20
//
unsigned __int8 *__fastcall sub_3401C20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __m128i a6)
{
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned __int8 *result; // rax
  unsigned __int8 *v10; // [rsp+0h] [rbp-80h]
  __int64 v11; // [rsp+20h] [rbp-60h] BYREF
  __int64 v12; // [rsp+28h] [rbp-58h]
  __int64 v13; // [rsp+30h] [rbp-50h] BYREF
  char v14; // [rsp+38h] [rbp-48h]
  unsigned __int64 v15; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v16; // [rsp+48h] [rbp-38h]
  __int64 v17; // [rsp+50h] [rbp-30h]
  __int64 v18; // [rsp+58h] [rbp-28h]

  v6 = (unsigned int)a5;
  v11 = a3;
  v12 = a4;
  if ( !BYTE4(a5) )
    return sub_3400BD0(a1, (unsigned int)a5, a2, (unsigned int)v11, a4, 0, a6, 0);
  if ( (_WORD)v11 )
  {
    if ( (_WORD)v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      BUG();
    v8 = 16LL * ((unsigned __int16)v11 - 1);
    v7 = *(_QWORD *)&byte_444C4A0[v8];
    LOBYTE(v8) = byte_444C4A0[v8 + 8];
  }
  else
  {
    v7 = sub_3007260((__int64)&v11);
    v17 = v7;
    v18 = v8;
  }
  v13 = v7;
  v14 = v8;
  v16 = sub_CA1930(&v13);
  if ( v16 > 0x40 )
    sub_C43690((__int64)&v15, v6, 0);
  else
    v15 = v6;
  result = sub_3401900(a1, a2, v11, v12, (__int64)&v15, 1, a6);
  if ( v16 > 0x40 )
  {
    if ( v15 )
    {
      v10 = result;
      j_j___libc_free_0_0(v15);
      return v10;
    }
  }
  return result;
}
