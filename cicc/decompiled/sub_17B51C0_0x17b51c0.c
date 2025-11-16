// Function: sub_17B51C0
// Address: 0x17b51c0
//
_QWORD *__fastcall sub_17B51C0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, double a5, double a6, double a7)
{
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  unsigned __int64 *v15; // r13
  __int64 v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rsi
  __int64 v19; // rsi
  unsigned __int8 *v20; // rsi
  unsigned __int8 *v21; // [rsp+8h] [rbp-58h] BYREF
  char v22[16]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v23; // [rsp+20h] [rbp-40h]

  if ( *(_BYTE *)(a3 + 16) > 0x10u )
    goto LABEL_7;
  v10 = (_QWORD *)a2;
  if ( sub_1593BB0(a3, a2, a3, (__int64)a4) )
    return v10;
  if ( *(_BYTE *)(a2 + 16) > 0x10u )
  {
LABEL_7:
    v23 = 257;
    v13 = sub_15FB440(27, (__int64 *)a2, a3, (__int64)v22, 0);
    v14 = a1[1];
    v10 = (_QWORD *)v13;
    if ( v14 )
    {
      v15 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v14 + 40, v13);
      v16 = v10[3];
      v17 = *v15;
      v10[4] = v15;
      v17 &= 0xFFFFFFFFFFFFFFF8LL;
      v10[3] = v17 | v16 & 7;
      *(_QWORD *)(v17 + 8) = v10 + 3;
      *v15 = *v15 & 7 | (unsigned __int64)(v10 + 3);
    }
    sub_164B780((__int64)v10, a4);
    v18 = *a1;
    if ( *a1 )
    {
      v21 = (unsigned __int8 *)*a1;
      sub_1623A60((__int64)&v21, v18, 2);
      v19 = v10[6];
      if ( v19 )
        sub_161E7C0((__int64)(v10 + 6), v19);
      v20 = v21;
      v10[6] = v21;
      if ( v20 )
        sub_1623210((__int64)&v21, v20, (__int64)(v10 + 6));
    }
  }
  else
  {
    v10 = (_QWORD *)sub_15A2D10((__int64 *)a2, a3, a5, a6, a7);
    v11 = sub_14DBA30((__int64)v10, a1[8], 0);
    if ( v11 )
      return (_QWORD *)v11;
  }
  return v10;
}
