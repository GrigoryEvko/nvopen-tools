// Function: sub_25F25C0
// Address: 0x25f25c0
//
unsigned __int8 *__fastcall sub_25F25C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 *a7)
{
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 *v12; // rsi
  __int64 v13; // rdx
  _BOOL8 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int16 v20; // ax
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned __int8 *v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned __int8 *v24[11]; // [rsp+18h] [rbp-58h] BYREF

  v23 = *(unsigned __int8 **)(a2 + 72);
  v10 = sub_29B77F0(a3, a4);
  v24[0] = (unsigned __int8 *)v10;
  if ( v10 )
  {
    v11 = *(_QWORD *)(*(_QWORD *)(v10 + 16) + 24LL);
    if ( (unsigned __int8)sub_DFA9E0(a6) )
    {
      v20 = *((_WORD *)v24[0] + 1) & 0xC00F;
      LOBYTE(v20) = v20 | 0x90;
      *((_WORD *)v24[0] + 1) = v20;
      *(_WORD *)(v11 + 2) = *(_WORD *)(v11 + 2) & 0xF003 | 0x24;
    }
    v12 = (__int64 *)sub_BD5C60(v11);
    *(_QWORD *)(v11 + 72) = sub_A7A090((__int64 *)(v11 + 72), v12, -1, 31);
    if ( (_BYTE)qword_4FF1BC8 )
    {
      sub_B31A00((__int64)v24[0], qword_4FF1AC8, qword_4FF1AD0);
    }
    else if ( (v23[35] & 4) != 0 )
    {
      v21 = sub_B31D10((__int64)v23, (__int64)v12, v13);
      sub_B31A00((__int64)v24[0], v21, v22);
    }
    v14 = a5 != 0;
    sub_25EFA30((__int64)v24[0], v14);
    v24[2] = (unsigned __int8 *)&v23;
    v24[3] = (unsigned __int8 *)v24;
    sub_25F1D20(a7, v14, v15, v16, v17, v18, a2, &v23, v24);
    return v24[0];
  }
  else
  {
    sub_25F21F0(a7, a2);
    return 0;
  }
}
