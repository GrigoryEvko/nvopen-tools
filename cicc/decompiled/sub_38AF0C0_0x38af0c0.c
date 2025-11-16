// Function: sub_38AF0C0
// Address: 0x38af0c0
//
__int64 __fastcall sub_38AF0C0(__int64 a1, __int64 *a2, __int64 *a3, int a4, double a5, double a6, double a7)
{
  unsigned __int64 v9; // r14
  unsigned int v10; // r12d
  char v12; // al
  __int64 *v13; // [rsp+0h] [rbp-60h] BYREF
  __int64 v14; // [rsp+8h] [rbp-58h] BYREF
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v16; // [rsp+20h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v13, a3, a5, a6, a7) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' in logical operation") )
    return 1;
  v10 = sub_38A1070((__int64 **)a1, *v13, &v14, a3, a5, a6, a7);
  if ( (_BYTE)v10 )
  {
    return 1;
  }
  else
  {
    v12 = *(_BYTE *)(*v13 + 8);
    if ( v12 == 16 )
      v12 = *(_BYTE *)(**(_QWORD **)(*v13 + 16) + 8LL);
    if ( v12 == 11 )
    {
      v16 = 257;
      *a2 = sub_15FB440(a4, v13, v14, (__int64)v15, 0);
    }
    else
    {
      v16 = 259;
      v15[0] = "instruction requires integer or integer vector operands";
      return (unsigned int)sub_38814C0(a1 + 8, v9, (__int64)v15);
    }
  }
  return v10;
}
