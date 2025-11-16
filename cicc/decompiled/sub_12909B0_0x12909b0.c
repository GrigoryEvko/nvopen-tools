// Function: sub_12909B0
// Address: 0x12909b0
//
_QWORD *__fastcall sub_12909B0(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r12
  __int64 v5; // rax
  __int64 v6; // rdi
  unsigned __int64 *v7; // r13
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rsi
  __int64 v12; // [rsp+8h] [rbp-48h] BYREF
  char v13[16]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v3 = (_QWORD *)a1[7];
  if ( v3 )
  {
    v3 = 0;
    if ( !sub_157EBA0(a1[7]) )
    {
      v14 = 257;
      v5 = sub_1648A60(56, 1);
      v3 = (_QWORD *)v5;
      if ( v5 )
        sub_15F8320(v5, a2, 0);
      v6 = a1[7];
      if ( v6 )
      {
        v7 = (unsigned __int64 *)a1[8];
        sub_157E9D0(v6 + 40, v3);
        v8 = v3[3];
        v9 = *v7;
        v3[4] = v7;
        v9 &= 0xFFFFFFFFFFFFFFF8LL;
        v3[3] = v9 | v8 & 7;
        *(_QWORD *)(v9 + 8) = v3 + 3;
        *v7 = *v7 & 7 | (unsigned __int64)(v3 + 3);
      }
      sub_164B780(v3, v13);
      v10 = a1[6];
      if ( v10 )
      {
        v12 = a1[6];
        sub_1623A60(&v12, v10, 2);
        if ( v3[6] )
          sub_161E7C0(v3 + 6);
        v11 = v12;
        v3[6] = v12;
        if ( v11 )
          sub_1623210(&v12, v11, v3 + 6);
      }
    }
  }
  a1[7] = 0;
  a1[8] = 0;
  return v3;
}
