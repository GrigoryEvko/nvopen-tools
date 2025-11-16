// Function: sub_31C4E20
// Address: 0x31c4e20
//
__int64 __fastcall sub_31C4E20(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v4; // rsi
  __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // r13
  unsigned __int8 *v8; // r14
  unsigned __int8 *v9; // r15
  unsigned __int8 *v10; // rbx
  _QWORD *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // rax
  __int64 v17; // [rsp+18h] [rbp-58h]
  __int64 v18; // [rsp+28h] [rbp-48h]
  __int64 v19; // [rsp+38h] [rbp-38h]

  v4 = a2 - a1;
  v5 = v4 >> 3;
  v18 = a1;
  if ( v4 > 0 )
  {
    do
    {
      v6 = *(_QWORD *)(v18 + 8 * (v5 >> 1));
      v17 = v18 + 8 * (v5 >> 1);
      v7 = *a3;
      v8 = *(unsigned __int8 **)(sub_318B6A0(*a3) + 16);
      v9 = *(unsigned __int8 **)(sub_318B6A0(v6) + 16);
      v10 = sub_98ACB0(v8, 6u);
      if ( v10 == sub_98ACB0(v9, 6u)
        && (v11 = (_QWORD *)sub_B2BE50(*a4),
            v12 = sub_BCB2B0(v11),
            v13 = sub_B43CA0(*(_QWORD *)(v7 + 16)),
            v19 = sub_D35010(v12, (__int64)v8, v12, (__int64)v9, v13 + 312, (__int64)a4, 0, 0),
            BYTE4(v19))
        && (int)v19 > 0 )
      {
        v5 >>= 1;
      }
      else
      {
        v5 = v5 - (v5 >> 1) - 1;
        v18 = v17 + 8;
      }
    }
    while ( v5 > 0 );
  }
  return v18;
}
