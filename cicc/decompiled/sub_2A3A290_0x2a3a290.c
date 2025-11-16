// Function: sub_2A3A290
// Address: 0x2a3a290
//
__int64 __fastcall sub_2A3A290(__int64 *a1, unsigned __int8 a2)
{
  __int16 v3; // cx
  unsigned __int64 v4; // rdx
  unsigned __int8 v5; // al
  __int64 v6; // rax
  _QWORD *v7; // r13
  __int64 result; // rax
  __int64 v9; // r12
  __int64 v10; // r15
  bool v11; // zf
  __int64 v12; // rax
  _QWORD *v13; // r14
  __int64 *v14; // rax
  _QWORD *v15; // rax
  _QWORD *v16; // rdi
  __int64 *v17; // r15
  unsigned int v18; // edx
  unsigned int v19; // r13d
  _QWORD *v20; // rax
  __int64 v21; // r14
  __int64 v22; // r13
  unsigned __int64 v23; // rcx
  __int16 v24; // dx
  __int16 v25; // ax
  __int64 v26; // rdi
  _QWORD *v27; // rax
  __int64 v28; // rdx
  _QWORD *v29; // rsi
  __int64 v30; // [rsp+0h] [rbp-70h]
  __int64 v31; // [rsp+0h] [rbp-70h]
  _QWORD v32[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v33; // [rsp+30h] [rbp-40h]

  v3 = *(_WORD *)(*a1 + 2);
  _BitScanReverse64(&v4, 1LL << v3);
  v5 = 63 - (v4 ^ 0x3F);
  if ( a2 > v5 )
    v5 = a2;
  *(_WORD *)(*a1 + 2) = v3 & 0xFFC0 | v5;
  v6 = sub_B43CB0(*a1);
  v7 = (_QWORD *)sub_B2BE50(v6);
  result = sub_2A3A050(*a1);
  v9 = (result + (1LL << a2) - 1) & -(1LL << a2);
  if ( result != v9 )
  {
    v10 = result;
    v11 = (unsigned __int8)sub_B4CE70(*a1) == 0;
    v12 = *a1;
    if ( v11 )
    {
      v13 = *(_QWORD **)(v12 + 72);
    }
    else
    {
      v28 = *(_QWORD *)(v12 - 32);
      v29 = *(_QWORD **)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) > 0x40u )
        v29 = (_QWORD *)*v29;
      v13 = sub_BCD420(*(__int64 **)(v12 + 72), (__int64)v29);
    }
    v14 = (__int64 *)sub_BCB2B0(v7);
    v15 = sub_BCD420(v14, v9 - v10);
    v16 = (_QWORD *)*v13;
    v32[0] = v13;
    v32[1] = v15;
    v17 = sub_BD0B90(v16, v32, 2, 0);
    v30 = *a1 + 24;
    v18 = *(_DWORD *)(*(_QWORD *)(*a1 + 8) + 8LL);
    v33 = 257;
    v19 = v18 >> 8;
    v20 = sub_BD2C40(80, 1u);
    v21 = (__int64)v20;
    if ( v20 )
      sub_B4CDD0((__int64)v20, v17, v19, 0, (__int64)v32, 0, v30, 0);
    v22 = v21;
    sub_BD6B90((unsigned __int8 *)v21, (unsigned __int8 *)*a1);
    _BitScanReverse64(&v23, 1LL << *(_WORD *)(*a1 + 2));
    v24 = (63 - (v23 ^ 0x3F)) | *(_WORD *)(v21 + 2) & 0xFFC0;
    *(_WORD *)(v21 + 2) = v24;
    v25 = v24 & 0xFFBF | *(_WORD *)(*a1 + 2) & 0x40;
    *(_WORD *)(v21 + 2) = v25;
    LOBYTE(v25) = v25 & 0x7F;
    *(_WORD *)(v21 + 2) = *(_WORD *)(*a1 + 2) & 0x80 | v25;
    sub_B47C00(v21, *a1, 0, 0);
    v26 = *a1;
    v31 = *(_QWORD *)(*a1 + 8);
    if ( *(_QWORD *)(v21 + 8) != v31 )
    {
      v33 = 257;
      v27 = sub_BD2C40(72, 1u);
      v22 = (__int64)v27;
      if ( v27 )
        sub_B51BF0((__int64)v27, v21, v31, (__int64)v32, v26 + 24, 0);
      v26 = *a1;
    }
    sub_BD84D0(v26, v22);
    result = sub_B43D60((_QWORD *)*a1);
    *a1 = v21;
  }
  return result;
}
