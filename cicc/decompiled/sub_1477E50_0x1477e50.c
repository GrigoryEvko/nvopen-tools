// Function: sub_1477E50
// Address: 0x1477e50
//
__int64 __fastcall sub_1477E50(__int64 a1, _DWORD *a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned int v5; // ebx
  unsigned int v6; // r15d
  __int64 *v7; // rax
  __int64 v8; // rax
  char v9; // cl
  unsigned int v10; // eax
  __int64 v11; // r15
  unsigned int v13; // r15d
  __int64 *v14; // rax
  __int64 v15; // rax
  char v16; // cl
  unsigned int v17; // eax
  unsigned __int64 v18; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-58h]
  unsigned __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-48h]
  unsigned __int64 v22; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-38h]

  v4 = sub_1456040(a1);
  v5 = sub_1456C90(a3, v4);
  if ( !(unsigned __int8)sub_1477C30(a3, a1) )
  {
    v11 = 0;
    if ( !(unsigned __int8)sub_1477B50(a3, a1) )
      return v11;
    *a2 = 38;
    v13 = v5 - 1;
    v14 = sub_1477920(a3, a1, 1u);
    sub_158ACE0(&v20, v14);
    v19 = v5;
    v15 = ~(1LL << ((unsigned __int8)v5 - 1));
    if ( v5 <= 0x40 )
    {
      v18 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
    }
    else
    {
      sub_16A4EF0(&v18, -1, 1);
      v15 = ~(1LL << v13);
      if ( v19 > 0x40 )
      {
        *(_QWORD *)(v18 + 8LL * (v13 >> 6)) &= ~(1LL << v13);
        v16 = v21;
        if ( v21 <= 0x40 )
        {
LABEL_20:
          v20 = ~v20 & (0xFFFFFFFFFFFFFFFFLL >> -v16);
LABEL_21:
          sub_16A7400(&v20);
          sub_16A7200(&v20, &v18);
          v17 = v21;
          v21 = 0;
          v23 = v17;
          v22 = v20;
          v11 = sub_145CF40(a3, (__int64)&v22);
          sub_135E100((__int64 *)&v22);
          sub_135E100((__int64 *)&v18);
          sub_135E100((__int64 *)&v20);
          return v11;
        }
LABEL_27:
        sub_16A8F40(&v20);
        goto LABEL_21;
      }
    }
    v16 = v21;
    v18 &= v15;
    if ( v21 <= 0x40 )
      goto LABEL_20;
    goto LABEL_27;
  }
  *a2 = 40;
  v6 = v5 - 1;
  v7 = sub_1477920(a3, a1, 1u);
  sub_158ABC0(&v20, v7);
  v19 = v5;
  v8 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 <= 0x40 )
  {
    v18 = 0;
LABEL_23:
    v9 = v21;
    v18 |= v8;
    if ( v21 <= 0x40 )
      goto LABEL_5;
    goto LABEL_24;
  }
  sub_16A4EF0(&v18, 0, 0);
  v8 = 1LL << v6;
  if ( v19 <= 0x40 )
    goto LABEL_23;
  *(_QWORD *)(v18 + 8LL * (v6 >> 6)) |= 1LL << v6;
  v9 = v21;
  if ( v21 <= 0x40 )
  {
LABEL_5:
    v20 = ~v20 & (0xFFFFFFFFFFFFFFFFLL >> -v9);
    goto LABEL_6;
  }
LABEL_24:
  sub_16A8F40(&v20);
LABEL_6:
  sub_16A7400(&v20);
  sub_16A7200(&v20, &v18);
  v10 = v21;
  v21 = 0;
  v23 = v10;
  v22 = v20;
  v11 = sub_145CF40(a3, (__int64)&v22);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return v11;
}
