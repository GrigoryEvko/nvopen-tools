// Function: sub_140D0B0
// Address: 0x140d0b0
//
_QWORD *__fastcall sub_140D0B0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, unsigned __int8 a6)
{
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rsi
  unsigned __int64 *v19; // [rsp+8h] [rbp-68h]
  __int64 v20; // [rsp+18h] [rbp-58h] BYREF
  char v21[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v22; // [rsp+30h] [rbp-40h]

  if ( *(_BYTE *)(a2 + 16) > 0x10u || *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v22 = 257;
    v13 = sub_15FB440(15, a2, a3, v21, 0);
    v14 = a1[1];
    v10 = (_QWORD *)v13;
    if ( v14 )
    {
      v19 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v14 + 40, v13);
      v15 = *v19;
      v16 = v10[3] & 7LL;
      v10[4] = v19;
      v15 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v15 | v16;
      *(_QWORD *)(v15 + 8) = v10 + 3;
      *v19 = *v19 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780(v10, a4);
    v17 = *a1;
    if ( *a1 )
    {
      v20 = *a1;
      sub_1623A60(&v20, v17, 2);
      if ( v10[6] )
        sub_161E7C0(v10 + 6);
      v18 = v20;
      v10[6] = v20;
      if ( v18 )
        sub_1623210(&v20, v18, v10 + 6);
    }
    if ( a5 )
      sub_15F2310(v10, 1);
    if ( a6 )
      sub_15F2330(v10, 1);
  }
  else
  {
    v10 = (_QWORD *)sub_15A2C20(a2, a3, a5, a6);
    v11 = sub_14DBA30(v10, a1[8], 0);
    if ( v11 )
      return (_QWORD *)v11;
  }
  return v10;
}
