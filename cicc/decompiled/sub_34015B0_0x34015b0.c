// Function: sub_34015B0
// Address: 0x34015b0
//
unsigned __int8 *__fastcall sub_34015B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        __m128i a7)
{
  unsigned __int16 v9; // bx
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int8 *result; // rax
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  unsigned __int8 *v17; // [rsp+0h] [rbp-70h]
  __int64 v18; // [rsp+10h] [rbp-60h] BYREF
  __int64 v19; // [rsp+18h] [rbp-58h]
  unsigned __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  __int64 v21; // [rsp+28h] [rbp-48h]

  v9 = a3;
  v18 = a3;
  v19 = a4;
  if ( !(_WORD)a3 )
  {
    if ( sub_30070B0((__int64)&v18) )
    {
      v9 = sub_3009970((__int64)&v18, a2, v14, v15, v16);
LABEL_4:
      LOWORD(v20) = v9;
      v21 = v10;
      if ( !v9 )
        goto LABEL_5;
LABEL_14:
      if ( v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
        BUG();
      v11 = *(_QWORD *)&byte_444C4A0[16 * v9 - 16];
      LODWORD(v21) = v11;
      if ( (unsigned int)v11 <= 0x40 )
        goto LABEL_6;
LABEL_17:
      sub_C43690((__int64)&v20, -1, 1);
      goto LABEL_9;
    }
LABEL_3:
    v10 = v19;
    goto LABEL_4;
  }
  if ( (unsigned __int16)(a3 - 17) > 0xD3u )
    goto LABEL_3;
  v21 = 0;
  v9 = word_4456580[(unsigned __int16)a3 - 1];
  LOWORD(v20) = v9;
  if ( v9 )
    goto LABEL_14;
LABEL_5:
  LODWORD(v11) = sub_3007260((__int64)&v20);
  LODWORD(v21) = v11;
  if ( (unsigned int)v11 > 0x40 )
    goto LABEL_17;
LABEL_6:
  v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v11;
  if ( !(_DWORD)v11 )
    v12 = 0;
  v20 = v12;
LABEL_9:
  result = sub_34007B0(a1, (__int64)&v20, a2, v18, v19, a5, a7, a6);
  if ( (unsigned int)v21 > 0x40 )
  {
    if ( v20 )
    {
      v17 = result;
      j_j___libc_free_0_0(v20);
      return v17;
    }
  }
  return result;
}
