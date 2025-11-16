// Function: sub_38AC790
// Address: 0x38ac790
//
__int64 __fastcall sub_38AC790(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int8 v7; // bl
  int v8; // eax
  char v9; // r14
  const char *v11; // rax
  _QWORD *v12; // rbx
  unsigned __int64 v13; // [rsp+8h] [rbp-88h]
  unsigned __int64 v14; // [rsp+18h] [rbp-78h]
  char v15; // [rsp+26h] [rbp-6Ah] BYREF
  char v16; // [rsp+27h] [rbp-69h] BYREF
  unsigned int v17; // [rsp+28h] [rbp-68h] BYREF
  int v18; // [rsp+2Ch] [rbp-64h] BYREF
  __int64 v19; // [rsp+30h] [rbp-60h] BYREF
  __int64 v20; // [rsp+38h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v22; // [rsp+50h] [rbp-40h]

  v7 = 0;
  v8 = *(_DWORD *)(a1 + 64);
  v17 = 0;
  v15 = 0;
  v18 = 0;
  v16 = 1;
  if ( v8 == 67 )
  {
    v7 = 1;
    v8 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v8;
  }
  v9 = 0;
  if ( v8 == 66 )
  {
    v9 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v14 = *(_QWORD *)(a1 + 56);
  v21[0] = "expected type";
  v22 = 259;
  if ( (unsigned __int8)sub_3891B00(a1, &v20, (__int64)v21, 0) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected comma after load's type") )
    return 1;
  v13 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v19, a3, a4, a5, a6)
    || (unsigned __int8)sub_388CFF0(a1, v7, &v16, &v18)
    || (unsigned __int8)sub_388CB60(a1, &v17, &v15) )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)v19 + 8LL) != 15 || !*(_BYTE *)(v20 + 8) || *(_BYTE *)(v20 + 8) == 12 )
  {
    HIBYTE(v22) = 1;
    v11 = "load operand must be a pointer to a first class type";
LABEL_20:
    v21[0] = v11;
    LOBYTE(v22) = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v13, (__int64)v21);
  }
  if ( v7 && !v17 )
  {
    HIBYTE(v22) = 1;
    v11 = "atomic load must have explicit non-zero alignment";
    goto LABEL_20;
  }
  if ( (unsigned int)(v18 - 5) <= 1 )
  {
    HIBYTE(v22) = 1;
    v11 = "atomic load cannot use Release ordering";
    goto LABEL_20;
  }
  if ( v20 == *(_QWORD *)(*(_QWORD *)v19 + 24LL) )
  {
    v22 = 257;
    v12 = sub_1648A60(64, 1u);
    if ( v12 )
      sub_15F8F80((__int64)v12, v20, v19, (__int64)v21, v9 & 1, v17, v18, v16, 0);
    *a2 = v12;
    return 2 * (unsigned int)(v15 != 0);
  }
  else
  {
    v22 = 259;
    v21[0] = "explicit pointee type doesn't match operand's pointee type";
    return (unsigned __int8)sub_38814C0(a1 + 8, v14, (__int64)v21);
  }
}
