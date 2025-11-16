// Function: sub_3022060
// Address: 0x3022060
//
__int64 __fastcall sub_3022060(_BYTE *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // al
  const char *v8; // r13
  unsigned __int8 *v9; // rsi
  size_t v10; // rdx
  _QWORD *v11; // rax
  __int64 result; // rax
  const char *v13; // rsi
  unsigned __int64 v14; // rdi
  char *v15; // rax
  const char *v16; // r13
  unsigned int v17; // ecx
  __int64 v18; // rsi
  __int64 v19; // rdx
  const char *v20; // rcx
  __int64 v21; // rax
  __int64 v22; // rdx
  __m128i v23; // [rsp+0h] [rbp-A0h] BYREF
  const char *v24; // [rsp+10h] [rbp-90h]
  __int64 v25; // [rsp+18h] [rbp-88h]
  __int16 v26; // [rsp+20h] [rbp-80h]
  __m128i v27[2]; // [rsp+30h] [rbp-70h] BYREF
  char v28; // [rsp+50h] [rbp-50h]
  char v29; // [rsp+51h] [rbp-4Fh]
  __m128i v30[4]; // [rsp+60h] [rbp-40h] BYREF

  v7 = a1[32] & 0xF;
  if ( v7 )
  {
    if ( v7 == 6 )
    {
      v29 = 1;
      v20 = byte_3F871B3;
      v27[0].m128i_i64[0] = (__int64)"' has unsupported appending linkage type";
      v21 = 0;
      v28 = 3;
      if ( (a1[7] & 0x10) != 0 )
      {
        v20 = sub_BD5D20((__int64)a1);
        v21 = v22;
      }
      v26 = 1283;
      v23.m128i_i64[0] = (__int64)"Symbol '";
      v24 = v20;
      v25 = v21;
      sub_9C6370(v30, &v23, v27, (__int64)v20, a5, a6);
      sub_C64D30((__int64)v30, 1u);
    }
    result = (v7 + 9) & 0xF;
    if ( (unsigned __int8)result > 1u )
    {
      v19 = *(_QWORD *)(a2 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v19) <= 5 )
      {
        v10 = 6;
        v9 = ".weak ";
        return sub_CB6200(a2, v9, v10);
      }
      *(_DWORD *)v19 = 1634039598;
      *(_WORD *)(v19 + 4) = 8299;
      *(_QWORD *)(a2 + 32) += 6LL;
      return 8299;
    }
  }
  else if ( *a1 == 3 )
  {
    v8 = ".visible ";
    if ( sub_B2FC80((__int64)a1) )
      v8 = ".extern ";
    v9 = (unsigned __int8 *)v8;
    v10 = strlen(v8);
    v11 = *(_QWORD **)(a2 + 32);
    if ( v10 > *(_QWORD *)(a2 + 24) - (_QWORD)v11 )
      return sub_CB6200(a2, v9, v10);
    v14 = (unsigned __int64)(v11 + 1) & 0xFFFFFFFFFFFFFFF8LL;
    *v11 = *(_QWORD *)v8;
    *(_QWORD *)((char *)v11 + (unsigned int)v10 - 8) = *(_QWORD *)&v8[(unsigned int)v10 - 8];
    v15 = (char *)v11 - v14;
    v16 = (const char *)(v8 - v15);
    result = ((_DWORD)v10 + (_DWORD)v15) & 0xFFFFFFF8;
    if ( (unsigned int)result >= 8 )
    {
      result = (unsigned int)result & 0xFFFFFFF8;
      v17 = 0;
      do
      {
        v18 = v17;
        v17 += 8;
        *(_QWORD *)(v14 + v18) = *(_QWORD *)&v16[v18];
      }
      while ( v17 < (unsigned int)result );
    }
    *(_QWORD *)(a2 + 32) += v10;
  }
  else
  {
    v13 = ".extern ";
    if ( !sub_B2FC80((__int64)a1) )
      v13 = ".visible ";
    return sub_904010(a2, v13);
  }
  return result;
}
