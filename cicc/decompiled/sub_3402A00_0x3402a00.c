// Function: sub_3402A00
// Address: 0x3402a00
//
unsigned __int8 *__fastcall sub_3402A00(
        _QWORD *a1,
        unsigned __int64 *a2,
        __int64 a3,
        __int64 a4,
        __m128i a5,
        __int64 a6,
        __int64 a7)
{
  unsigned __int16 v7; // bx
  __int64 v8; // rdx
  unsigned __int8 *result; // rax
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  unsigned __int8 *v13; // [rsp+0h] [rbp-60h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h]
  unsigned __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  __int64 v17; // [rsp+28h] [rbp-38h]

  v7 = a3;
  v14 = a3;
  v15 = a4;
  if ( !(_WORD)a3 )
  {
    if ( sub_30070B0((__int64)&v14) )
    {
      v7 = sub_3009970((__int64)&v14, (__int64)a2, v10, v11, v12);
LABEL_4:
      LOWORD(v16) = v7;
      v17 = v8;
      if ( !v7 )
        goto LABEL_5;
LABEL_12:
      if ( v7 == 1 || (unsigned __int16)(v7 - 504) <= 7u )
        BUG();
      LODWORD(v17) = *(_QWORD *)&byte_444C4A0[16 * v7 - 16];
      if ( (unsigned int)v17 <= 0x40 )
        goto LABEL_6;
      goto LABEL_15;
    }
LABEL_3:
    v8 = v15;
    goto LABEL_4;
  }
  if ( (unsigned __int16)(a3 - 17) > 0xD3u )
    goto LABEL_3;
  v17 = 0;
  v7 = word_4456580[(unsigned __int16)a3 - 1];
  LOWORD(v16) = v7;
  if ( v7 )
    goto LABEL_12;
LABEL_5:
  LODWORD(v17) = sub_3007260((__int64)&v16);
  if ( (unsigned int)v17 <= 0x40 )
  {
LABEL_6:
    v16 = 1;
    goto LABEL_7;
  }
LABEL_15:
  sub_C43690((__int64)&v16, 1, 0);
LABEL_7:
  result = sub_3402600(a1, a2, (unsigned int)v14, v15, (__int64)&v16, a7, a5);
  if ( (unsigned int)v17 > 0x40 )
  {
    if ( v16 )
    {
      v13 = result;
      j_j___libc_free_0_0(v16);
      return v13;
    }
  }
  return result;
}
