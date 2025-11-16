// Function: sub_21D6C00
// Address: 0x21d6c00
//
__int64 __fastcall sub_21D6C00(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, __m128i a7)
{
  char v9; // al
  __int64 v10; // rdx
  unsigned int v12; // ebx
  __int64 v13; // rax
  unsigned int v14; // [rsp+8h] [rbp-48h]
  unsigned int v15; // [rsp+10h] [rbp-40h] BYREF
  __int64 v16; // [rsp+18h] [rbp-38h]

  v9 = *(_BYTE *)(a2 + 88);
  v10 = *(_QWORD *)(a2 + 96);
  LOBYTE(v15) = v9;
  v16 = v10;
  if ( v9 == 2 )
    return sub_21D6AB0(a5, a6, *(double *)a7.m128i_i64, (__int64)a1, a2, a3, a4);
  if ( v9 != 86 )
  {
    if ( v9 )
    {
      if ( (unsigned __int8)(v9 - 14) > 0x5Fu )
        return 0;
    }
    else if ( !sub_1F58D20((__int64)&v15) )
    {
      return 0;
    }
    return sub_21D62E0(a5, a6, a7, (__int64)a1, a2, a3, (__int64)a4);
  }
  v14 = sub_1E34390(*(_QWORD *)(a2 + 104));
  v12 = sub_1E340A0(*(_QWORD *)(a2 + 104));
  v13 = sub_1E0A0C0(a4[4]);
  if ( (unsigned __int8)sub_1F43CC0((__int64)a1, a4[6], v13, v15, v16, v12, v14, 0) )
    return sub_21D62E0(a5, a6, a7, (__int64)a1, a2, a3, (__int64)a4);
  return sub_20BB820(a1, (_QWORD *)a2, a4, a5, a6, a7);
}
