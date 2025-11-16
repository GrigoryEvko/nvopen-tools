// Function: sub_39A9340
// Address: 0x39a9340
//
__int64 __fastcall sub_39A9340(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __m128i *v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r15
  __int64 v10; // rdi
  __int64 v11; // rdx
  __m128i v13; // rax
  _QWORD v14[2]; // [rsp+0h] [rbp-80h] BYREF
  __int64 v15[2]; // [rsp+10h] [rbp-70h] BYREF
  _DWORD v16[4]; // [rsp+20h] [rbp-60h] BYREF
  __m128i v17; // [rsp+30h] [rbp-50h] BYREF
  char v18; // [rsp+40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 624);
  if ( !v2 )
    return sub_39CC330(*(_QWORD *)(a1 + 616), a2);
  if ( !*(_BYTE *)(a1 + 632) )
  {
    *(_BYTE *)(a1 + 632) = 1;
    sub_39A3640(a1, a1 + 8, 16, 0);
    v2 = *(_QWORD *)(a1 + 624);
    if ( !*(_BYTE *)(a2 + 56) )
      goto LABEL_4;
LABEL_11:
    v13.m128i_i64[0] = sub_161E970(*(_QWORD *)(a2 + 48));
    v18 = 1;
    v17 = v13;
    goto LABEL_5;
  }
  if ( *(_BYTE *)(a2 + 56) )
    goto LABEL_11;
LABEL_4:
  v18 = 0;
LABEL_5:
  v3 = sub_39A3100(a1, a2);
  v4 = *(unsigned int *)(a2 + 8);
  v5 = (__int64)v3;
  v6 = *(_QWORD *)(a2 - 8 * v4);
  if ( v6 )
  {
    v7 = sub_161E970(*(_QWORD *)(a2 - 8LL * *(unsigned int *)(a2 + 8)));
    v9 = v8;
    v4 = *(unsigned int *)(a2 + 8);
    v6 = v7;
  }
  else
  {
    v9 = 0;
  }
  v10 = *(_QWORD *)(a2 + 8 * (1 - v4));
  if ( v10 )
    v10 = sub_161E970(v10);
  else
    v11 = 0;
  v14[0] = v10;
  v14[1] = v11;
  v15[0] = v6;
  v15[1] = v9;
  sub_38C89A0((__int64)v16, v2, (__int64)v14, v15, v5, &v17, 0);
  return v16[0];
}
