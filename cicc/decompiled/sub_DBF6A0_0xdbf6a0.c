// Function: sub_DBF6A0
// Address: 0xdbf6a0
//
__int64 __fastcall sub_DBF6A0(__int64 a1, _DWORD *a2, __int64 *a3)
{
  __int64 v4; // rax
  unsigned int v5; // ebx
  unsigned int v6; // r14d
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned int v9; // edx
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // r14
  __int64 v14; // rax
  __int64 v15; // r8
  __int64 v16; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+18h] [rbp-58h]
  unsigned __int64 v18; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+28h] [rbp-48h]
  unsigned __int64 v20; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-38h]

  v4 = sub_D95540(a1);
  v5 = sub_D97050((__int64)a3, v4);
  if ( !(unsigned __int8)sub_DBEDC0((__int64)a3, a1) )
  {
    v12 = 0;
    if ( !(unsigned __int8)sub_DBEC00((__int64)a3, a1) )
      return v12;
    *a2 = 38;
    v14 = sub_DBB9F0((__int64)a3, a1, 1u, 0);
    sub_AB14C0((__int64)&v18, v14);
    v17 = v5;
    v15 = ~(1LL << ((unsigned __int8)v5 - 1));
    if ( v5 > 0x40 )
    {
      sub_C43690((__int64)&v16, -1, 1);
      v15 = ~(1LL << ((unsigned __int8)v5 - 1));
      if ( v17 > 0x40 )
      {
        *(_QWORD *)(v16 + 8LL * ((v5 - 1) >> 6)) &= ~(1LL << ((unsigned __int8)v5 - 1));
        goto LABEL_5;
      }
    }
    else
    {
      if ( v5 )
        v12 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v5;
      v16 = v12;
    }
    v16 &= v15;
    goto LABEL_5;
  }
  *a2 = 40;
  v6 = v5 - 1;
  v7 = sub_DBB9F0((__int64)a3, a1, 1u, 0);
  sub_AB13A0((__int64)&v18, v7);
  v17 = v5;
  v8 = 1LL << ((unsigned __int8)v5 - 1);
  if ( v5 > 0x40 )
  {
    sub_C43690((__int64)&v16, 0, 0);
    v8 = 1LL << v6;
    if ( v17 > 0x40 )
    {
      *(_QWORD *)(v16 + 8LL * (v6 >> 6)) |= 1LL << v6;
      v9 = v19;
      if ( v19 <= 0x40 )
        goto LABEL_6;
      goto LABEL_21;
    }
  }
  else
  {
    v16 = 0;
  }
  v16 |= v8;
LABEL_5:
  v9 = v19;
  if ( v19 <= 0x40 )
  {
LABEL_6:
    v10 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v9) & ~v18;
    if ( !v9 )
      v10 = 0;
    v18 = v10;
    goto LABEL_9;
  }
LABEL_21:
  sub_C43D10((__int64)&v18);
LABEL_9:
  sub_C46250((__int64)&v18);
  sub_C45EE0((__int64)&v18, &v16);
  v11 = v19;
  v19 = 0;
  v21 = v11;
  v20 = v18;
  v12 = (__int64)sub_DA26C0(a3, (__int64)&v20);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  if ( v19 > 0x40 && v18 )
    j_j___libc_free_0_0(v18);
  return v12;
}
