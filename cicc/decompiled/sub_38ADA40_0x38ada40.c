// Function: sub_38ADA40
// Address: 0x38ada40
//
__int64 __fastcall sub_38ADA40(__int64 a1, _QWORD *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned __int64 v6; // r15
  unsigned int v7; // r12d
  unsigned int *v9; // r15
  __int64 *v10; // r14
  _QWORD *v11; // rbx
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rax
  const char *v16; // rax
  __int64 v17; // [rsp+8h] [rbp-88h]
  char v18; // [rsp+17h] [rbp-79h] BYREF
  __int64 *v19; // [rsp+18h] [rbp-78h] BYREF
  _QWORD v20[2]; // [rsp+20h] [rbp-70h] BYREF
  __int16 v21; // [rsp+30h] [rbp-60h]
  unsigned int *v22; // [rsp+40h] [rbp-50h] BYREF
  __int64 v23; // [rsp+48h] [rbp-48h]
  _BYTE v24[64]; // [rsp+50h] [rbp-40h] BYREF

  v22 = (unsigned int *)v24;
  v6 = *(_QWORD *)(a1 + 56);
  v23 = 0x400000000LL;
  if ( (unsigned __int8)sub_38AB270((__int64 **)a1, &v19, a3, a4, a5, a6)
    || (unsigned __int8)sub_388D130(a1, (__int64)&v22, &v18) )
  {
    v7 = 1;
    goto LABEL_3;
  }
  if ( (unsigned int)*(unsigned __int8 *)(*v19 + 8) - 13 > 1 )
  {
    HIBYTE(v21) = 1;
    v16 = "extractvalue operand must be aggregate type";
LABEL_18:
    v20[0] = v16;
    LOBYTE(v21) = 3;
    v7 = (unsigned __int8)sub_38814C0(a1 + 8, v6, (__int64)v20);
    goto LABEL_3;
  }
  if ( !sub_15FB2A0(*v19, v22, (unsigned int)v23) )
  {
    HIBYTE(v21) = 1;
    v16 = "invalid indices for extractvalue";
    goto LABEL_18;
  }
  v9 = v22;
  v10 = v19;
  v21 = 257;
  v17 = (unsigned int)v23;
  v11 = sub_1648A60(88, 1u);
  if ( v11 )
  {
    v12 = sub_15FB2A0(*v10, v9, v17);
    sub_15F1EA0((__int64)v11, v12, 62, (__int64)(v11 - 3), 1, 0);
    if ( *(v11 - 3) )
    {
      v13 = *(v11 - 2);
      v14 = *(v11 - 1) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v14 = v13;
      if ( v13 )
        *(_QWORD *)(v13 + 16) = *(_QWORD *)(v13 + 16) & 3LL | v14;
    }
    *(v11 - 3) = v10;
    v15 = v10[1];
    *(v11 - 2) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = (unsigned __int64)(v11 - 2) | *(_QWORD *)(v15 + 16) & 3LL;
    *(v11 - 1) = (unsigned __int64)(v10 + 1) | *(v11 - 1) & 3LL;
    v10[1] = (__int64)(v11 - 3);
    v11[7] = v11 + 9;
    v11[8] = 0x400000000LL;
    sub_15FB110((__int64)v11, v9, v17, (__int64)v20);
  }
  *a2 = v11;
  v7 = 2 * (v18 != 0);
LABEL_3:
  if ( v22 != (unsigned int *)v24 )
    _libc_free((unsigned __int64)v22);
  return v7;
}
