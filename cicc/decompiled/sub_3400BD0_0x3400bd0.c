// Function: sub_3400BD0
// Address: 0x3400bd0
//
unsigned __int8 *__fastcall sub_3400BD0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        unsigned __int8 a8)
{
  unsigned __int16 v10; // bx
  __int64 v11; // rdx
  unsigned __int8 *result; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  unsigned __int8 *v16; // [rsp+10h] [rbp-70h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h]
  unsigned __int64 v19; // [rsp+30h] [rbp-50h] BYREF
  __int64 v20; // [rsp+38h] [rbp-48h]

  v10 = a4;
  v17 = a4;
  v18 = a5;
  if ( !(_WORD)a4 )
  {
    if ( sub_30070B0((__int64)&v17) )
    {
      v10 = sub_3009970((__int64)&v17, a2, v13, v14, v15);
LABEL_4:
      LOWORD(v19) = v10;
      v20 = v11;
      if ( !v10 )
        goto LABEL_5;
LABEL_12:
      if ( v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
        BUG();
      LODWORD(v20) = *(_QWORD *)&byte_444C4A0[16 * v10 - 16];
      if ( (unsigned int)v20 <= 0x40 )
        goto LABEL_6;
      goto LABEL_15;
    }
LABEL_3:
    v11 = v18;
    goto LABEL_4;
  }
  if ( (unsigned __int16)(a4 - 17) > 0xD3u )
    goto LABEL_3;
  v20 = 0;
  v10 = word_4456580[(unsigned __int16)a4 - 1];
  LOWORD(v19) = v10;
  if ( v10 )
    goto LABEL_12;
LABEL_5:
  LODWORD(v20) = sub_3007260((__int64)&v19);
  if ( (unsigned int)v20 <= 0x40 )
  {
LABEL_6:
    v19 = a2;
    goto LABEL_7;
  }
LABEL_15:
  sub_C43690((__int64)&v19, a2, 0);
LABEL_7:
  result = sub_34007B0(a1, (__int64)&v19, a3, v17, v18, a6, a7, a8);
  if ( (unsigned int)v20 > 0x40 )
  {
    if ( v19 )
    {
      v16 = result;
      j_j___libc_free_0_0(v19);
      return v16;
    }
  }
  return result;
}
