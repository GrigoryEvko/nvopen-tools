// Function: sub_31DE730
// Address: 0x31de730
//
__int64 __fastcall sub_31DE730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rsi
  char v8; // al
  __int64 v9; // rdx
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rdi
  __int64 *v13; // rax
  __m128i v14; // rax
  __int64 v15; // rcx
  __int64 v16; // r8
  __int64 v17; // r9
  __m128i v18; // [rsp+0h] [rbp-110h] BYREF
  __int16 v19; // [rsp+20h] [rbp-F0h]
  __m128i v20[2]; // [rsp+30h] [rbp-E0h] BYREF
  char v21; // [rsp+50h] [rbp-C0h]
  char v22; // [rsp+51h] [rbp-BFh]
  __m128i v23[3]; // [rsp+60h] [rbp-B0h] BYREF
  __m128i v24[2]; // [rsp+90h] [rbp-80h] BYREF
  char v25; // [rsp+B0h] [rbp-60h]
  char v26; // [rsp+B1h] [rbp-5Fh]
  __m128i v27[5]; // [rsp+C0h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(a1 + 280);
  if ( (*(_BYTE *)(v7 + 8) & 4) != 0 )
  {
    v8 = *(_BYTE *)(v7 + 9);
    if ( (v8 & 0x70) == 0x20 )
    {
      *(_QWORD *)(v7 + 24) = 0;
      *(_BYTE *)(v7 + 9) = v8 & 0x8F;
    }
    *(_BYTE *)(v7 + 8) &= ~4u;
    *(_QWORD *)v7 = 0;
    v7 = *(_QWORD *)(a1 + 280);
  }
  if ( (*(_BYTE *)(v7 + 9) & 0x70) == 0x20 )
  {
    v26 = 1;
    v24[0].m128i_i64[0] = (__int64)"' is a protected alias";
    v25 = 3;
    if ( (*(_BYTE *)(v7 + 8) & 1) != 0 )
    {
      v13 = *(__int64 **)(v7 - 8);
      v14.m128i_i64[1] = *v13;
      v14.m128i_i64[0] = (__int64)(v13 + 3);
    }
    else
    {
      v14 = 0u;
    }
    v18 = v14;
    v20[0].m128i_i64[0] = (__int64)"'";
    v19 = 261;
    v22 = 1;
    v21 = 3;
    sub_9C6370(v23, v20, &v18, a4, a5, a6);
    sub_9C6370(v27, v23, v24, v15, v16, v17);
    sub_C64D30((__int64)v27, 1u);
  }
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a1 + 224) + 208LL))(*(_QWORD *)(a1 + 224), v7, 0);
  result = *(_QWORD *)(a1 + 200);
  if ( *(_DWORD *)(result + 564) == 3 )
  {
    result = sub_31DE680(a1, **(_QWORD **)(a1 + 232), v9);
    v11 = result;
    if ( *(_QWORD *)(a1 + 280) != result )
    {
      v12 = *(_QWORD *)(a1 + 224);
      *(_QWORD *)(a1 + 544) = result;
      (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v12 + 208LL))(v12, result, 0);
      return (*(__int64 (__fastcall **)(_QWORD, __int64, __int64))(**(_QWORD **)(a1 + 224) + 296LL))(
               *(_QWORD *)(a1 + 224),
               v11,
               2);
    }
  }
  return result;
}
