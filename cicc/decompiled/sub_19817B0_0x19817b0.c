// Function: sub_19817B0
// Address: 0x19817b0
//
__int64 __fastcall sub_19817B0(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // r15
  __int64 v12; // rax
  __int64 v13; // r15
  _QWORD *v14; // r13
  _QWORD *v15; // rax
  __int64 v16; // r14
  _QWORD **v17; // rax
  _QWORD *v18; // rbx
  __int64 *v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdi
  unsigned __int64 *v22; // rbx
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __int64 v25; // rsi
  __int64 v26; // rsi
  unsigned __int8 *v27; // rsi
  __int64 v29; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v30; // [rsp+28h] [rbp-78h] BYREF
  __int64 v31; // [rsp+30h] [rbp-70h] BYREF
  __int16 v32; // [rsp+40h] [rbp-60h]
  char v33[16]; // [rsp+50h] [rbp-50h] BYREF
  __int16 v34; // [rsp+60h] [rbp-40h]

  v10 = sub_1456040(a5);
  if ( (unsigned __int8)sub_148B410(*a1, a1[2], a4, a5, a6) )
    return sub_159C4F0(*(__int64 **)(a3 + 24));
  v29 = sub_38767A0(a2, a5, v10, a7);
  v12 = sub_38767A0(a2, a6, v10, a7);
  v32 = 257;
  v13 = v12;
  if ( *(_BYTE *)(v29 + 16) <= 0x10u && *(_BYTE *)(v12 + 16) <= 0x10u )
    return sub_15A37B0(a4, (_QWORD *)v29, (_QWORD *)v12, 0);
  v34 = 257;
  v15 = sub_1648A60(56, 2u);
  v14 = v15;
  if ( v15 )
  {
    v16 = (__int64)v15;
    v17 = *(_QWORD ***)v29;
    if ( *(_BYTE *)(*(_QWORD *)v29 + 8LL) == 16 )
    {
      v18 = v17[4];
      v19 = (__int64 *)sub_1643320(*v17);
      v20 = (__int64)sub_16463B0(v19, (unsigned int)v18);
    }
    else
    {
      v20 = sub_1643320(*v17);
    }
    sub_15FEC10((__int64)v14, v20, 51, a4, v29, v13, (__int64)v33, 0);
  }
  else
  {
    v16 = 0;
  }
  v21 = *(_QWORD *)(a3 + 8);
  if ( v21 )
  {
    v22 = *(unsigned __int64 **)(a3 + 16);
    sub_157E9D0(v21 + 40, (__int64)v14);
    v23 = v14[3];
    v24 = *v22;
    v14[4] = v22;
    v24 &= 0xFFFFFFFFFFFFFFF8LL;
    v14[3] = v24 | v23 & 7;
    *(_QWORD *)(v24 + 8) = v14 + 3;
    *v22 = *v22 & 7 | (unsigned __int64)(v14 + 3);
  }
  sub_164B780(v16, &v31);
  v25 = *(_QWORD *)a3;
  if ( *(_QWORD *)a3 )
  {
    v30 = *(unsigned __int8 **)a3;
    sub_1623A60((__int64)&v30, v25, 2);
    v26 = v14[6];
    if ( v26 )
      sub_161E7C0((__int64)(v14 + 6), v26);
    v27 = v30;
    v14[6] = v30;
    if ( v27 )
      sub_1623210((__int64)&v30, v27, (__int64)(v14 + 6));
  }
  return (__int64)v14;
}
