// Function: sub_3024560
// Address: 0x3024560
//
unsigned __int64 __fastcall sub_3024560(
        __int64 a1,
        unsigned __int32 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // rbp
  __int64 v7; // r12
  __int64 v9; // rdx
  unsigned __int64 result; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // rdx
  __m128i v14; // [rsp-A8h] [rbp-A8h] BYREF
  __int16 v15; // [rsp-88h] [rbp-88h]
  __m128i v16; // [rsp-78h] [rbp-78h] BYREF
  char v17; // [rsp-58h] [rbp-58h]
  char v18; // [rsp-57h] [rbp-57h]
  __m128i v19[3]; // [rsp-48h] [rbp-48h] BYREF
  __int64 v20; // [rsp-10h] [rbp-10h]
  __int64 v21; // [rsp-8h] [rbp-8h]

  if ( a2 == 4 )
  {
    v13 = *(_QWORD *)(a3 + 32);
    result = *(_QWORD *)(a3 + 24) - v13;
    if ( result <= 4 )
    {
      return sub_CB6200(a3, (unsigned __int8 *)"const", 5u);
    }
    else
    {
      *(_DWORD *)v13 = 1936617315;
      *(_BYTE *)(v13 + 4) = 116;
      *(_QWORD *)(a3 + 32) += 5LL;
    }
  }
  else if ( a2 > 4 )
  {
    if ( a2 != 5 )
      goto LABEL_18;
    v11 = *(_QWORD *)(a3 + 32);
    result = *(_QWORD *)(a3 + 24) - v11;
    if ( result <= 4 )
    {
      return sub_CB6200(a3, (unsigned __int8 *)"local", 5u);
    }
    else
    {
      *(_DWORD *)v11 = 1633906540;
      *(_BYTE *)(v11 + 4) = 108;
      *(_QWORD *)(a3 + 32) += 5LL;
    }
  }
  else
  {
    if ( a2 != 1 )
    {
      if ( a2 == 3 )
      {
        v9 = *(_QWORD *)(a3 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(a3 + 24) - v9) <= 5 )
          return sub_CB6200(a3, (unsigned __int8 *)"shared", 6u);
        *(_DWORD *)v9 = 1918986355;
        *(_WORD *)(v9 + 4) = 25701;
        *(_QWORD *)(a3 + 32) += 6LL;
        return 25701;
      }
LABEL_18:
      v21 = v6;
      v20 = v7;
      v14.m128i_i32[0] = a2;
      v15 = 265;
      v18 = 1;
      v16.m128i_i64[0] = (__int64)"Bad address space found while emitting PTX: ";
      v17 = 3;
      sub_9C6370(v19, &v16, &v14, a4, a5, a6);
      sub_C64D30((__int64)v19, 1u);
    }
    v12 = *(_QWORD *)(a3 + 32);
    result = *(_QWORD *)(a3 + 24) - v12;
    if ( result <= 5 )
    {
      return sub_CB6200(a3, (unsigned __int8 *)"global", 6u);
    }
    else
    {
      *(_DWORD *)v12 = 1651469415;
      *(_WORD *)(v12 + 4) = 27745;
      *(_QWORD *)(a3 + 32) += 6LL;
    }
  }
  return result;
}
