// Function: sub_1632080
// Address: 0x1632080
//
__int64 __fastcall sub_1632080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 ***v7; // rax
  __int64 **v8; // rbx
  __int64 ***v9; // r12
  __int64 **v10; // rax
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rcx
  __int64 v15; // rax
  _QWORD v16[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v17; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  v16[0] = a2;
  v16[1] = a3;
  v7 = (__int64 ***)sub_1632000(a1, a2, a3);
  if ( v7 )
  {
    v8 = *v7;
    v9 = v7;
    if ( v8 != (__int64 **)sub_1646BA0(a4, 0) )
    {
      v10 = (__int64 **)sub_1646BA0(a4, 0);
      return sub_15A4510(v9, v10, 0);
    }
    return (__int64)v9;
  }
  else
  {
    v18 = 261;
    v17 = v16;
    v12 = sub_1648B60(120);
    v13 = v12;
    if ( v12 )
      sub_15E2490(v12, a4, 0, (__int64)&v17, 0);
    if ( (*(_BYTE *)(v13 + 33) & 0x20) == 0 )
      *(_QWORD *)(v13 + 112) = a5;
    sub_1631B60(a1 + 24, v13);
    v14 = *(_QWORD *)(a1 + 24);
    v15 = *(_QWORD *)(v13 + 56);
    *(_QWORD *)(v13 + 64) = a1 + 24;
    v14 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(v13 + 56) = v14 | v15 & 7;
    *(_QWORD *)(v14 + 8) = v13 + 56;
    *(_QWORD *)(a1 + 24) = *(_QWORD *)(a1 + 24) & 7LL | (v13 + 56);
    return v13;
  }
}
