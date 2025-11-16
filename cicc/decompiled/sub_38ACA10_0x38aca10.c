// Function: sub_38ACA10
// Address: 0x38aca10
//
__int64 __fastcall sub_38ACA10(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int8 v6; // r13
  int v8; // eax
  unsigned __int8 v9; // r14
  unsigned __int64 v10; // r15
  const char *v12; // rax
  _QWORD *v13; // r12
  unsigned __int64 v14; // [rsp+0h] [rbp-80h]
  char v15; // [rsp+16h] [rbp-6Ah] BYREF
  char v16; // [rsp+17h] [rbp-69h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-68h] BYREF
  int v18; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v19; // [rsp+20h] [rbp-60h] BYREF
  __int64 v20; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v21[2]; // [rsp+30h] [rbp-50h] BYREF
  char v22; // [rsp+40h] [rbp-40h]
  char v23; // [rsp+41h] [rbp-3Fh]

  v6 = 0;
  v8 = *(_DWORD *)(a1 + 64);
  v17 = 0;
  v15 = 0;
  v18 = 0;
  v16 = 1;
  if ( v8 == 67 )
  {
    v6 = 1;
    v8 = sub_3887100(a1 + 8);
    *(_DWORD *)(a1 + 64) = v8;
  }
  v9 = 0;
  if ( v8 == 66 )
  {
    v9 = 1;
    *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  }
  v10 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v19, a3, a4, a5, a6) )
    return 1;
  if ( (unsigned __int8)sub_388AF10(a1, 4, "expected ',' after store operand") )
    return 1;
  v14 = *(_QWORD *)(a1 + 56);
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v20, a3, a4, a5, a6)
    || (unsigned __int8)sub_388CFF0(a1, v6, &v16, &v18)
    || (unsigned __int8)sub_388CB60(a1, &v17, &v15) )
  {
    return 1;
  }
  if ( *(_BYTE *)(*(_QWORD *)v20 + 8LL) != 15 )
  {
    v23 = 1;
    v22 = 3;
    v21[0] = "store operand must be a pointer";
    return (unsigned __int8)sub_38814C0(a1 + 8, v14, (__int64)v21);
  }
  if ( !*(_BYTE *)(*(_QWORD *)v19 + 8LL) || *(_BYTE *)(*(_QWORD *)v19 + 8LL) == 12 )
  {
    v23 = 1;
    v12 = "store operand must be a first class value";
    goto LABEL_17;
  }
  if ( *(_QWORD *)v19 != *(_QWORD *)(*(_QWORD *)v20 + 24LL) )
  {
    v23 = 1;
    v12 = "stored value and pointer type do not match";
LABEL_17:
    v21[0] = v12;
    v22 = 3;
    return (unsigned __int8)sub_38814C0(a1 + 8, v10, (__int64)v21);
  }
  if ( v6 && !v17 )
  {
    v23 = 1;
    v12 = "atomic store must have explicit non-zero alignment";
    goto LABEL_17;
  }
  if ( (v18 & 0xFFFFFFFD) == 4 )
  {
    v23 = 1;
    v12 = "atomic store cannot use Acquire ordering";
    goto LABEL_17;
  }
  v13 = sub_1648A60(64, 2u);
  if ( v13 )
    sub_15F9480((__int64)v13, v19, v20, v9, v17, v18, v16, 0);
  *a2 = v13;
  return 2 * (unsigned int)(v15 != 0);
}
