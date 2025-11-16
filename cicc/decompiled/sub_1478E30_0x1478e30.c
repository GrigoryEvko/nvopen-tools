// Function: sub_1478E30
// Address: 0x1478e30
//
__int64 __fastcall sub_1478E30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r14
  __int64 v5; // rbx
  __int16 v6; // ax
  __int64 v7; // r9
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rsi
  unsigned int v12; // eax
  __int64 v13; // r14
  __int64 v14; // rsi
  __int64 v15; // [rsp+8h] [rbp-78h]
  int v16; // [rsp+1Ch] [rbp-64h] BYREF
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  __int64 v18; // [rsp+28h] [rbp-58h] BYREF
  __int64 v19; // [rsp+30h] [rbp-50h] BYREF
  __int64 v20; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v21; // [rsp+40h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+48h] [rbp-38h]

  v4 = a3;
  v5 = a4;
  v6 = *(_WORD *)(a4 + 24);
  if ( v6 != 7 )
  {
LABEL_2:
    if ( !v6 && !*(_WORD *)(v4 + 24) )
    {
      v10 = *(_QWORD *)(v5 + 32) + 24LL;
      sub_13A38D0((__int64)&v21, *(_QWORD *)(v4 + 32) + 24LL);
      v11 = v10;
      goto LABEL_22;
    }
LABEL_4:
    v17 = 0;
    v18 = 0;
    v19 = 0;
    v20 = 0;
    if ( !(unsigned __int8)sub_1457900(a2, v5, &v17, &v18, &v16) || (v7 = v17, *(_WORD *)(v17 + 24)) )
    {
      v7 = 0;
    }
    else if ( v18 == v4 )
    {
      sub_13A38D0((__int64)&v21, *(_QWORD *)(v17 + 32) + 24LL);
      if ( v22 <= 0x40 )
        v21 = ~v21 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v22);
      else
        sub_16A8F40(&v21);
      sub_16A7400(&v21);
      goto LABEL_23;
    }
    v15 = v7;
    if ( !(unsigned __int8)sub_1457900(a2, v4, &v19, &v20, &v16) || *(_WORD *)(v19 + 24) )
      goto LABEL_12;
    if ( v20 == v5 )
    {
      v14 = *(_QWORD *)(v19 + 32);
      *(_BYTE *)(a1 + 16) = 1;
      sub_13A38D0(a1, v14 + 24);
      return a1;
    }
    if ( !v15 || v20 != v18 )
      goto LABEL_12;
    v13 = *(_QWORD *)(v15 + 32) + 24LL;
    sub_13A38D0((__int64)&v21, *(_QWORD *)(v19 + 32) + 24LL);
    v11 = v13;
LABEL_22:
    sub_16A7590(&v21, v11);
LABEL_23:
    v12 = v22;
    *(_BYTE *)(a1 + 16) = 1;
    v22 = 0;
    *(_DWORD *)(a1 + 8) = v12;
    *(_QWORD *)a1 = v21;
    sub_135E100((__int64 *)&v21);
    return a1;
  }
  if ( *(_WORD *)(a3 + 24) != 7 )
    goto LABEL_4;
  if ( *(_QWORD *)(a4 + 48) == *(_QWORD *)(a3 + 48) && *(_QWORD *)(a4 + 40) == 2 && *(_QWORD *)(a3 + 40) == 2 )
  {
    v9 = sub_13A5BC0((_QWORD *)a4, a2);
    if ( v9 == sub_13A5BC0((_QWORD *)v4, a2) )
    {
      v5 = **(_QWORD **)(v5 + 32);
      v4 = **(_QWORD **)(v4 + 32);
      v6 = *(_WORD *)(v5 + 24);
      goto LABEL_2;
    }
  }
LABEL_12:
  *(_BYTE *)(a1 + 16) = 0;
  return a1;
}
