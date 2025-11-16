// Function: sub_156D390
// Address: 0x156d390
//
__int64 __fastcall sub_156D390(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v6; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 *v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // rcx
  __int64 v14; // rsi
  __int64 v15; // rsi
  __int64 v16; // [rsp+8h] [rbp-48h] BYREF
  char v17[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v18; // [rsp+20h] [rbp-30h]

  v6 = (_QWORD *)a2;
  if ( *(_BYTE *)(a3 + 16) > 0x10u )
  {
LABEL_5:
    v18 = 257;
    v9 = sub_15FB440(27, a2, a3, v17, 0);
    v10 = a1[1];
    v6 = (_QWORD *)v9;
    if ( v10 )
    {
      v11 = (unsigned __int64 *)a1[2];
      sub_157E9D0(v10 + 40, v9);
      v12 = v6[3];
      v13 = *v11;
      v6[4] = v11;
      v13 &= 0xFFFFFFFFFFFFFFF8LL;
      v6[3] = v13 | v12 & 7;
      *(_QWORD *)(v13 + 8) = v6 + 3;
      *v11 = *v11 & 7 | (unsigned __int64)(v6 + 3);
    }
    sub_164B780(v6, a4);
    v14 = *a1;
    if ( *a1 )
    {
      v16 = *a1;
      sub_1623A60(&v16, v14, 2);
      if ( v6[6] )
        sub_161E7C0(v6 + 6);
      v15 = v16;
      v6[6] = v16;
      if ( v15 )
        sub_1623210(&v16, v15, v6 + 6);
    }
    return (__int64)v6;
  }
  if ( !(unsigned __int8)sub_1593BB0(a3) )
  {
    if ( *(_BYTE *)(a2 + 16) <= 0x10u )
      return sub_15A2D10(a2, a3);
    goto LABEL_5;
  }
  return (__int64)v6;
}
