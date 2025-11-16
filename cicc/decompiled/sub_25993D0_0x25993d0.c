// Function: sub_25993D0
// Address: 0x25993d0
//
_BOOL8 __fastcall sub_25993D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r13
  unsigned __int8 v5; // al
  unsigned __int64 v6; // rsi
  char v7; // dl
  __int64 v8; // r15
  unsigned __int64 v9; // rdi
  unsigned __int8 v10; // r10
  char v11; // r15
  unsigned __int8 v12; // r15
  _BOOL8 result; // rax
  __int64 v14; // rax
  char v15; // dl
  char v16; // r10
  unsigned __int8 v17; // dl
  unsigned __int8 *v18; // rax
  unsigned __int64 v19; // rax
  __int64 v20; // [rsp+8h] [rbp-58h]
  unsigned __int8 v21; // [rsp+8h] [rbp-58h]
  bool v22; // [rsp+17h] [rbp-49h] BYREF
  __int64 v23; // [rsp+18h] [rbp-48h] BYREF
  __int64 v24; // [rsp+20h] [rbp-40h] BYREF
  __int64 v25; // [rsp+28h] [rbp-38h]

  v2 = (__int64 *)(a1 + 72);
  v5 = sub_2509800((_QWORD *)(a1 + 72));
  if ( v5 <= 7u && ((1LL << v5) & 0xA8) != 0 )
  {
    v6 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
    if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
      v6 = *(_QWORD *)(v6 + 24);
    v7 = 5;
  }
  else
  {
    v18 = sub_250CBE0(v2, a2);
    v7 = 4;
    v6 = (unsigned __int64)v18;
  }
  sub_250D230((unsigned __int64 *)&v24, v6, v7, 0);
  v8 = v24;
  v20 = v25;
  v9 = sub_250C680(v2);
  if ( v9 && (unsigned __int8)sub_B2D680(v9) || (v14 = sub_25294B0(a2, v8, v20, a1, 1, 0, 1)) == 0 )
  {
    v10 = *(_BYTE *)(a1 + 97);
    v11 = 0;
  }
  else
  {
    v15 = *(_BYTE *)(v14 + 96);
    v16 = *(_BYTE *)(a1 + 97);
    v11 = *(_BYTE *)(v14 + 97);
    *(_BYTE *)(a1 + 96) |= v15;
    v10 = v15 | v16;
    *(_BYTE *)(a1 + 97) = v10;
    v17 = *(_BYTE *)(v14 + 97);
    result = 1;
    if ( (v10 & v17) == v10 )
      return result;
  }
  v21 = v10;
  v23 = 0;
  if ( (unsigned __int8)sub_25890A0(a2, a1, v2, 1, &v22, 0, &v23) || v23 && (*(_WORD *)(v23 + 98) & 3) == 3 )
  {
    v24 = a2;
    v25 = a1;
    v19 = sub_250D070(v2);
    if ( (unsigned __int8)sub_252FFB0(
                            a2,
                            (unsigned __int8 (__fastcall *)(__int64, __int64, __int64 *))sub_2598340,
                            (__int64)&v24,
                            a1,
                            v19,
                            0,
                            1,
                            1,
                            0,
                            0) )
    {
      return *(_BYTE *)(a1 + 97) == v21;
    }
    else
    {
      *(_BYTE *)(a1 + 97) = *(_BYTE *)(a1 + 96);
      return 0;
    }
  }
  else
  {
    v12 = *(_BYTE *)(a1 + 96) | *(_BYTE *)(a1 + 97) & v11;
    *(_BYTE *)(a1 + 97) = v12;
    return v21 == v12;
  }
}
