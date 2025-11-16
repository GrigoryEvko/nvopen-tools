// Function: sub_256F8A0
// Address: 0x256f8a0
//
_BOOL8 __fastcall sub_256F8A0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rax
  __int64 v3; // rbx
  unsigned __int64 v4; // r12
  _BOOL8 result; // rax
  _QWORD *v6; // rax
  __int64 *v7; // r15
  __int64 *v8; // rax
  __int64 v9; // rbx
  unsigned __int64 v10; // rax
  __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rbx
  __int64 v14; // [rsp+0h] [rbp-80h]
  __int64 v15; // [rsp+10h] [rbp-70h]
  unsigned int v16; // [rsp+18h] [rbp-68h]
  unsigned __int8 v17; // [rsp+1Ch] [rbp-64h]
  __int64 v18; // [rsp+20h] [rbp-60h] BYREF
  __int64 v19; // [rsp+28h] [rbp-58h]
  __int16 v20; // [rsp+40h] [rbp-40h]

  v2 = sub_2509740((_QWORD *)(a1 + 72));
  v3 = *(_QWORD *)(a1 + 104);
  v4 = v2;
  result = 1;
  if ( *(_BYTE *)v4 == 60 )
  {
    v6 = (_QWORD *)sub_BD5C60(v4);
    v18 = (unsigned __int64)(v3 + 7) >> 3;
    LODWORD(v19) = 32;
    v7 = (__int64 *)sub_BCB2B0(v6);
    v8 = (__int64 *)sub_BD5C60(v4);
    v15 = sub_ACCFD0(v8, (__int64)&v18);
    sub_969240(&v18);
    v9 = *(_QWORD *)(v4 + 32);
    _BitScanReverse64(&v10, 1LL << *(_WORD *)(v4 + 2));
    v16 = *(_DWORD *)(*(_QWORD *)(v4 + 8) + 8LL) >> 8;
    v17 = 63 - (v10 ^ 0x3F);
    v18 = (__int64)sub_BD5D20(v4);
    v14 = v9;
    v20 = 261;
    v19 = v11;
    v12 = (unsigned __int8 *)sub_BD2C40(80, unk_3F10A14);
    v13 = v12;
    if ( v12 )
      sub_B4CCA0((__int64)v12, v7, v16, v15, v17, (__int64)&v18, v14, 0);
    sub_250D230((unsigned __int64 *)&v18, v4, 1, 0);
    return (unsigned __int8)sub_256F570(a2, v18, v19, v13, 1u) == 0;
  }
  return result;
}
