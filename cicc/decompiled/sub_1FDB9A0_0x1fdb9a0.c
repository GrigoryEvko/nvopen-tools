// Function: sub_1FDB9A0
// Address: 0x1fdb9a0
//
__int64 __fastcall sub_1FDB9A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 result; // rax
  unsigned int v4; // r8d
  __int64 v5; // rax
  __int64 *v6; // r14
  __int64 v7; // r9
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rcx
  int v15; // r8d
  __int64 v16; // rax
  unsigned int v17; // [rsp+4h] [rbp-6Ch]
  __int64 v18; // [rsp+8h] [rbp-68h]
  __m128i v19; // [rsp+10h] [rbp-60h] BYREF
  __int64 v20; // [rsp+20h] [rbp-50h]
  __int64 v21; // [rsp+28h] [rbp-48h]
  int v22; // [rsp+30h] [rbp-40h]

  v2 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v2 + 16) == 20 )
  {
    if ( *(_BYTE *)(v2 + 96) )
    {
      sub_1FD5930((__int64)a1);
      result = 0;
      if ( *(_QWORD *)(v2 + 64) )
        return result;
    }
    else
    {
      result = 0;
      if ( *(_QWORD *)(v2 + 64) )
        return result;
    }
    v4 = *(unsigned __int8 *)(v2 + 96);
    if ( *(_BYTE *)(v2 + 97) )
      v4 |= 2u;
    v5 = a1[5];
    v17 = v4;
    v6 = *(__int64 **)(v5 + 792);
    v7 = *(_QWORD *)(v5 + 784);
    v8 = *(_QWORD *)(v7 + 56);
    v18 = v7;
    v9 = (__int64)sub_1E0B640(v8, *(_QWORD *)(a1[13] + 8) + 64LL, a1 + 10, 0);
    sub_1DD5BA0((__int64 *)(v18 + 16), v9);
    v10 = *v6;
    v11 = *(_QWORD *)v9;
    *(_QWORD *)(v9 + 8) = v6;
    v10 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v9 = v10 | v11 & 7;
    *(_QWORD *)(v10 + 8) = v9;
    *v6 = v9 | *v6 & 7;
    v12 = *(_QWORD *)(v2 + 24);
    v19.m128i_i8[0] = 9;
    v21 = v12;
    v19.m128i_i32[0] &= 0xFFF000FF;
    v20 = 0;
    v19.m128i_i32[2] = 0;
    v22 = 0;
    sub_1E1A9C0(v9, v8, &v19);
    v19.m128i_i64[0] = 1;
    v21 = v17;
    v20 = 0;
    sub_1E1A9C0(v9, v8, &v19);
    return 1;
  }
  else
  {
    sub_1E2DAF0(a2, *(_QWORD *)(*(_QWORD *)(a1[5] + 8) + 32LL));
    v16 = *(_QWORD *)(a2 - 24);
    if ( *(_BYTE *)(v16 + 16) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
    {
      sub_1FD5930((__int64)a1);
      return sub_1FD74F0(a1, a2);
    }
    else
    {
      return sub_1FDB330(a1, a2, v13, v14, v15);
    }
  }
}
