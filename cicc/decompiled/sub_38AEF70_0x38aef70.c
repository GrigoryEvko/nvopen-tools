// Function: sub_38AEF70
// Address: 0x38aef70
//
__int64 __fastcall sub_38AEF70(__int64 a1, __int64 *a2, __int64 *a3, int a4, int a5, double a6, double a7, double a8)
{
  unsigned __int64 v11; // r15
  unsigned int v12; // r9d
  __int64 v14; // rdx
  char v15; // al
  __int64 v16; // rax
  __int64 *v17; // [rsp+10h] [rbp-60h] BYREF
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  _QWORD v19[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v11 = *(_QWORD *)(a1 + 56);
  if ( !(unsigned __int8)sub_38AB270((__int64 **)a1, &v17, a3, a6, a7, a8)
    && !(unsigned __int8)sub_388AF10(a1, 4, "expected ',' in arithmetic operation")
    && !(unsigned __int8)sub_38A1070((__int64 **)a1, *v17, &v18, a3, a6, a7, a8) )
  {
    v14 = *v17;
    v15 = *(_BYTE *)(*v17 + 8);
    if ( a5 == 1 )
    {
      if ( v15 == 16 )
        v15 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
      if ( v15 == 11 )
        goto LABEL_12;
    }
    else
    {
      if ( a5 == 2 )
      {
        if ( v15 == 16 )
          v15 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
      }
      else
      {
        if ( v15 == 16 )
          v15 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
        if ( v15 == 11 )
          goto LABEL_12;
      }
      if ( (unsigned __int8)(v15 - 1) <= 5u )
      {
LABEL_12:
        v20 = 257;
        v16 = sub_15FB440(a4, v17, v18, (__int64)v19, 0);
        v12 = 0;
        *a2 = v16;
        return v12;
      }
    }
    v20 = 259;
    v19[0] = "invalid operand type for instruction";
    return (unsigned int)sub_38814C0(a1 + 8, v11, (__int64)v19);
  }
  return 1;
}
