// Function: sub_17B5310
// Address: 0x17b5310
//
_QWORD *__fastcall sub_17B5310(__int64 *a1, __int16 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  _QWORD *v8; // r12
  __int64 v9; // rax
  _QWORD *v11; // rax
  __int64 v12; // r9
  _QWORD **v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rsi
  __int64 v22; // rsi
  unsigned __int8 *v23; // rsi
  __int64 v24; // [rsp+8h] [rbp-78h]
  _QWORD *v25; // [rsp+10h] [rbp-70h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v29; // [rsp+28h] [rbp-58h] BYREF
  char v30[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v31; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v31 = 257;
    v11 = sub_1648A60(56, 2u);
    v12 = a4;
    v8 = v11;
    if ( v11 )
    {
      v28 = (__int64)v11;
      v13 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v24 = v12;
        v25 = v13[4];
        v14 = (__int64 *)sub_1643320(*v13);
        v15 = (__int64)sub_16463B0(v14, (unsigned int)v25);
        v16 = v24;
      }
      else
      {
        v26 = v12;
        v15 = sub_1643320(*v13);
        v16 = v26;
      }
      sub_15FEC10((__int64)v8, v15, 51, a2, a3, v16, (__int64)v30, 0);
    }
    else
    {
      v28 = 0;
    }
    v17 = a1[1];
    if ( v17 )
    {
      v18 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v17 + 40, (__int64)v8);
      v19 = v8[3];
      v20 = *v18;
      v8[4] = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      v8[3] = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v8 + 3;
      *v18 = *v18 & 7 | (unsigned __int64)(v8 + 3);
    }
    sub_164B780(v28, a5);
    v21 = *a1;
    if ( *a1 )
    {
      v29 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v29, v21, 2);
      v22 = v8[6];
      if ( v22 )
        sub_161E7C0((__int64)(v8 + 6), v22);
      v23 = v29;
      v8[6] = v29;
      if ( v23 )
        sub_1623210((__int64)&v29, v23, (__int64)(v8 + 6));
    }
  }
  else
  {
    v8 = (_QWORD *)sub_15A37B0(a2, (_QWORD *)a3, (_QWORD *)a4, 0);
    v9 = sub_14DBA30((__int64)v8, a1[8], 0);
    if ( v9 )
      return (_QWORD *)v9;
  }
  return v8;
}
