// Function: sub_1790840
// Address: 0x1790840
//
unsigned __int8 *__fastcall sub_1790840(__int64 a1, __int16 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  unsigned __int8 *v8; // rax
  unsigned __int8 *v9; // r14
  __int64 v10; // r15
  _QWORD **v11; // rax
  __int64 *v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdi
  unsigned __int64 *v15; // r12
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rdx
  bool v19; // zf
  __int64 v20; // rsi
  __int64 v21; // rsi
  unsigned __int8 *v22; // rsi
  _QWORD *v24; // [rsp+8h] [rbp-78h]
  unsigned __int8 *v27; // [rsp+28h] [rbp-58h] BYREF
  char v28[16]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v29; // [rsp+40h] [rbp-40h]

  v29 = 257;
  v8 = (unsigned __int8 *)sub_1648A60(56, 2u);
  v9 = v8;
  if ( v8 )
  {
    v10 = (__int64)v8;
    v11 = *(_QWORD ***)a3;
    if ( *(_BYTE *)(*(_QWORD *)a3 + 8LL) == 16 )
    {
      v24 = v11[4];
      v12 = (__int64 *)sub_1643320(*v11);
      v13 = (__int64)sub_16463B0(v12, (unsigned int)v24);
    }
    else
    {
      v13 = sub_1643320(*v11);
    }
    sub_15FEC10((__int64)v9, v13, 51, a2, a3, a4, (__int64)v28, 0);
  }
  else
  {
    v10 = 0;
  }
  v14 = *(_QWORD *)(a1 + 8);
  if ( v14 )
  {
    v15 = *(unsigned __int64 **)(a1 + 16);
    sub_157E9D0(v14 + 40, (__int64)v9);
    v16 = *((_QWORD *)v9 + 3);
    v17 = *v15;
    *((_QWORD *)v9 + 4) = v15;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    *((_QWORD *)v9 + 3) = v17 | v16 & 7;
    *(_QWORD *)(v17 + 8) = v9 + 24;
    *v15 = *v15 & 7 | (unsigned __int64)(v9 + 24);
  }
  sub_164B780(v10, a5);
  v19 = *(_QWORD *)(a1 + 80) == 0;
  v27 = v9;
  if ( v19 )
    sub_4263D6(v10, a5, v18);
  (*(void (__fastcall **)(__int64, unsigned __int8 **))(a1 + 88))(a1 + 64, &v27);
  v20 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    v27 = *(unsigned __int8 **)a1;
    sub_1623A60((__int64)&v27, v20, 2);
    v21 = *((_QWORD *)v9 + 6);
    if ( v21 )
      sub_161E7C0((__int64)(v9 + 48), v21);
    v22 = v27;
    *((_QWORD *)v9 + 6) = v27;
    if ( v22 )
      sub_1623210((__int64)&v27, v22, (__int64)(v9 + 48));
  }
  return v9;
}
