// Function: sub_17203D0
// Address: 0x17203d0
//
unsigned __int8 *__fastcall sub_17203D0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned __int8 *v8; // r12
  __int64 v9; // rax
  unsigned __int8 *v11; // rax
  __int64 v12; // r9
  _QWORD **v13; // rax
  __int64 *v14; // rax
  __int64 v15; // rax
  __int64 v16; // r9
  __int64 v17; // rdi
  unsigned __int64 *v18; // r13
  __int64 v19; // rax
  unsigned __int64 v20; // rcx
  __int64 v21; // rdx
  bool v22; // zf
  __int64 v23; // rsi
  __int64 v24; // rsi
  unsigned __int8 *v25; // rsi
  __int64 v26; // [rsp+8h] [rbp-78h]
  _QWORD *v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+10h] [rbp-70h]
  __int64 v30; // [rsp+18h] [rbp-68h]
  unsigned __int8 *v31; // [rsp+28h] [rbp-58h] BYREF
  char v32[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v33; // [rsp+40h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u || *(_BYTE *)(a4 + 16) > 0x10u )
  {
    v33 = 257;
    v11 = (unsigned __int8 *)sub_1648A60(56, 2u);
    v12 = a4;
    v8 = v11;
    if ( v11 )
    {
      v30 = (__int64)v11;
      v13 = *(_QWORD ***)a3;
      if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
      {
        v26 = v12;
        v27 = v13[4];
        v14 = (__int64 *)sub_1643320(*v13);
        v15 = (__int64)sub_16463B0(v14, (unsigned int)v27);
        v16 = v26;
      }
      else
      {
        v28 = v12;
        v15 = sub_1643320(*v13);
        v16 = v28;
      }
      sub_15FEC10((__int64)v8, v15, 51, a2, a3, v16, (__int64)v32, 0);
    }
    else
    {
      v30 = 0;
    }
    v17 = *(_QWORD *)(a1 + 8);
    if ( v17 )
    {
      v18 = *(unsigned __int64 **)(a1 + 16);
      sub_157E9D0(v17 + 40, (__int64)v8);
      v19 = *((_QWORD *)v8 + 3);
      v20 = *v18;
      *((_QWORD *)v8 + 4) = v18;
      v20 &= 0xFFFFFFFFFFFFFFF8LL;
      *((_QWORD *)v8 + 3) = v20 | v19 & 7;
      *(_QWORD *)(v20 + 8) = v8 + 24;
      *v18 = *v18 & 7 | (unsigned __int64)(v8 + 24);
    }
    sub_164B780(v30, a5);
    v22 = *(_QWORD *)(a1 + 80) == 0;
    v31 = v8;
    if ( v22 )
      sub_4263D6(v30, a5, v21);
    (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v31);
    v23 = *(_QWORD *)a1;
    if ( *(_QWORD *)a1 )
    {
      v31 = *(unsigned __int8 **)a1;
      sub_1623A60((__int64)&v31, v23, 2);
      v24 = *((_QWORD *)v8 + 6);
      if ( v24 )
        sub_161E7C0((__int64)(v8 + 48), v24);
      v25 = v31;
      *((_QWORD *)v8 + 6) = v31;
      if ( v25 )
        sub_1623210((__int64)&v31, v25, (__int64)(v8 + 48));
    }
  }
  else
  {
    v8 = (unsigned __int8 *)sub_15A37B0(a2, (_QWORD *)a3, (_QWORD *)a4, 0);
    v9 = sub_14DBA30((__int64)v8, *(_QWORD *)(a1 + 96), 0);
    if ( v9 )
      return (unsigned __int8 *)v9;
  }
  return v8;
}
