// Function: sub_38AE490
// Address: 0x38ae490
//
__int64 __fastcall sub_38AE490(__int64 a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v6; // r12d
  unsigned __int64 v8; // r14
  _QWORD *v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r14
  __int64 v12; // rcx
  unsigned __int64 v13; // rdx
  __int64 v14; // rdx
  _QWORD *v15; // [rsp+8h] [rbp-68h]
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  __int64 v17; // [rsp+18h] [rbp-58h] BYREF
  __int64 v18[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v19; // [rsp+30h] [rbp-40h]

  v17 = 0;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v16, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after vaarg operand") )
    return 1;
  v8 = *(_QWORD *)(a1 + 56);
  v18[0] = (__int64)"expected type";
  v19 = 259;
  v6 = sub_3891B00(a1, &v17, (__int64)v18, 0);
  if ( (_BYTE)v6 )
  {
    return 1;
  }
  else if ( *(_BYTE *)(v17 + 8) == 12 || !*(_BYTE *)(v17 + 8) )
  {
    v19 = 259;
    v18[0] = (__int64)"va_arg requires operand with first class type";
    return (unsigned int)sub_38814C0(a1 + 8, v8, (__int64)v18);
  }
  else
  {
    v19 = 257;
    v9 = sub_1648A60(56, 1u);
    v10 = (__int64)v9;
    if ( v9 )
    {
      v15 = v9 - 3;
      v11 = v16;
      sub_15F1EA0((__int64)v9, v17, 58, (__int64)(v9 - 3), 1, 0);
      if ( *(_QWORD *)(v10 - 24) )
      {
        v12 = *(_QWORD *)(v10 - 16);
        v13 = *(_QWORD *)(v10 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v13 = v12;
        if ( v12 )
          *(_QWORD *)(v12 + 16) = *(_QWORD *)(v12 + 16) & 3LL | v13;
      }
      *(_QWORD *)(v10 - 24) = v11;
      if ( v11 )
      {
        v14 = *(_QWORD *)(v11 + 8);
        *(_QWORD *)(v10 - 16) = v14;
        if ( v14 )
          *(_QWORD *)(v14 + 16) = (v10 - 16) | *(_QWORD *)(v14 + 16) & 3LL;
        *(_QWORD *)(v10 - 8) = (v11 + 8) | *(_QWORD *)(v10 - 8) & 3LL;
        *(_QWORD *)(v11 + 8) = v15;
      }
      sub_164B780(v10, v18);
    }
    *a2 = v10;
  }
  return v6;
}
