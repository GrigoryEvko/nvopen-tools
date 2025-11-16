// Function: sub_F50590
// Address: 0xf50590
//
__int64 __fastcall sub_F50590(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  char v3; // r14
  __int64 v4; // rax
  char v5; // dl
  char v6; // bl
  unsigned __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // r15d
  _BYTE *v14; // rax
  unsigned __int64 v15; // [rsp+0h] [rbp-50h] BYREF
  __int64 v16; // [rsp+8h] [rbp-48h]
  unsigned __int8 v17; // [rsp+10h] [rbp-40h]

  v2 = sub_B141B0(a2) + 312;
  v3 = sub_AE5020(v2, a1);
  v4 = sub_9208B0(v2, a1);
  v6 = v5;
  v7 = 8 * (((1LL << v3) + ((unsigned __int64)(v4 + 7) >> 3) - 1) >> v3 << v3);
  v8 = sub_B11F60(a2 + 80);
  v9 = sub_B12000(a2 + 72);
  v10 = sub_AF4940(v8, v9);
  v16 = v11;
  v12 = (unsigned __int8)v11;
  v15 = v10;
  if ( (_BYTE)v11
    || !*(_BYTE *)(a2 + 64)
    && (v14 = (_BYTE *)sub_B12A50(a2, 0)) != 0
    && *v14 == 60
    && (sub_B4CFC0((__int64)&v15, (__int64)v14, v2), (v12 = v17) != 0)
    && (v6 || (v12 = 0, !(_BYTE)v16)) )
  {
    LOBYTE(v12) = v15 <= v7;
  }
  return v12;
}
