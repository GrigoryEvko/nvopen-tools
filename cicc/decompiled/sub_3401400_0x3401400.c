// Function: sub_3401400
// Address: 0x3401400
//
unsigned __int8 *__fastcall sub_3401400(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 a6,
        __m128i a7,
        unsigned __int8 a8)
{
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned __int64 v12; // r15
  unsigned __int64 v13; // rsi
  unsigned __int8 *result; // rax
  bool v15; // al
  __int64 v16; // rdx
  __int64 v17; // r8
  unsigned int v18; // [rsp+Ch] [rbp-74h]
  unsigned __int8 *v19; // [rsp+10h] [rbp-70h]
  __int64 v20; // [rsp+20h] [rbp-60h] BYREF
  __int64 v21; // [rsp+28h] [rbp-58h]
  unsigned __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  __int64 v23; // [rsp+38h] [rbp-48h]

  v20 = a4;
  v21 = a5;
  if ( !(_WORD)a4 )
  {
    v18 = a4;
    v15 = sub_30070B0((__int64)&v20);
    LOWORD(a4) = v18;
    if ( v15 )
    {
      LOWORD(a4) = sub_3009970((__int64)&v20, a2, v16, v18, v17);
LABEL_4:
      LOWORD(v22) = a4;
      v23 = v10;
      if ( !(_WORD)a4 )
        goto LABEL_5;
LABEL_14:
      if ( (_WORD)a4 == 1 || (unsigned __int16)(a4 - 504) <= 7u )
        BUG();
      v11 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)a4 - 16];
      LODWORD(v23) = v11;
      if ( (unsigned int)v11 <= 0x40 )
        goto LABEL_6;
LABEL_17:
      sub_C43690((__int64)&v22, a2, 1);
      goto LABEL_9;
    }
LABEL_3:
    v10 = v21;
    goto LABEL_4;
  }
  if ( (unsigned __int16)(a4 - 17) > 0xD3u )
    goto LABEL_3;
  v23 = 0;
  LOWORD(a4) = word_4456580[(unsigned __int16)a4 - 1];
  LOWORD(v22) = a4;
  if ( (_WORD)a4 )
    goto LABEL_14;
LABEL_5:
  LODWORD(v11) = sub_3007260((__int64)&v22);
  LODWORD(v23) = v11;
  if ( (unsigned int)v11 > 0x40 )
    goto LABEL_17;
LABEL_6:
  v12 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v11) & a2;
  v13 = 0;
  if ( (_DWORD)v11 )
    v13 = v12;
  v22 = v13;
LABEL_9:
  result = sub_34007B0(a1, (__int64)&v22, a3, v20, v21, a6, a7, a8);
  if ( (unsigned int)v23 > 0x40 )
  {
    if ( v22 )
    {
      v19 = result;
      j_j___libc_free_0_0(v22);
      return v19;
    }
  }
  return result;
}
