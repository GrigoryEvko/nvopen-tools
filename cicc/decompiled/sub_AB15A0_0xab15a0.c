// Function: sub_AB15A0
// Address: 0xab15a0
//
__int64 __fastcall sub_AB15A0(__int64 a1, int a2, __int64 a3)
{
  unsigned int v4; // r14d
  unsigned int v5; // eax
  unsigned int v6; // eax
  __int64 *v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  unsigned int v11; // r8d
  unsigned int v12; // eax
  unsigned int v13; // eax
  unsigned int v14; // ebx
  unsigned int v15; // eax
  unsigned int v16; // ebx
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // [rsp+0h] [rbp-70h] BYREF
  unsigned int v22; // [rsp+8h] [rbp-68h]
  __int64 v23; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h] BYREF
  unsigned int v28; // [rsp+38h] [rbp-38h]

  if ( sub_AAF7D0(a3) )
  {
    v5 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a1 + 8) = v5;
    if ( v5 > 0x40 )
    {
      sub_C43780(a1, a3);
      v19 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a1 + 24) = v19;
      if ( v19 <= 0x40 )
        goto LABEL_5;
    }
    else
    {
      *(_QWORD *)a1 = *(_QWORD *)a3;
      v6 = *(_DWORD *)(a3 + 24);
      *(_DWORD *)(a1 + 24) = v6;
      if ( v6 <= 0x40 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
        return a1;
      }
    }
    sub_C43780(a1 + 16, a3 + 16);
    return a1;
  }
  v4 = *(_DWORD *)(a3 + 8);
  switch ( a2 )
  {
    case ' ':
      sub_AAF450(a1, a3);
      return a1;
    case '!':
      if ( sub_9876C0((__int64 *)a3) )
      {
        sub_9865C0((__int64)&v27, a3);
        sub_9865C0((__int64)&v25, a3 + 16);
        sub_AADC30(a1, (__int64)&v25, &v27);
        sub_969240(&v25);
        sub_969240(&v27);
      }
      else
      {
        sub_AADB10(a1, v4, 1);
      }
      return a1;
    case '"':
      v8 = &v21;
      sub_AB0A00((__int64)&v21, a3);
      v14 = v22;
      if ( !v22 )
        goto LABEL_23;
      if ( v22 > 0x40 )
      {
        if ( v14 == (unsigned int)sub_C445E0(&v21) )
          goto LABEL_23;
        goto LABEL_18;
      }
      if ( v21 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v22) )
      {
LABEL_18:
        sub_9691E0((__int64)&v27, v4, 0, 0, 0);
        goto LABEL_29;
      }
LABEL_23:
      sub_AADB10(a1, v4, 0);
LABEL_21:
      sub_969240(v8);
      break;
    case '#':
      v8 = &v27;
      sub_9691E0((__int64)&v27, v4, 0, 0, 0);
      sub_AB0A00((__int64)&v25, a3);
      goto LABEL_20;
    case '$':
      v8 = &v23;
      sub_AB0910((__int64)&v23, a3);
      if ( sub_9867B0((__int64)&v23) )
        goto LABEL_23;
      v20 = v24;
      v24 = 0;
      v28 = v20;
      v27 = v23;
      sub_9691E0((__int64)&v25, v4, 0, 0, 0);
      goto LABEL_9;
    case '%':
      sub_AB0910((__int64)&v25, a3);
      sub_C46A40(&v25, 1);
      v15 = v26;
      v26 = 0;
      v28 = v15;
      v27 = v25;
      sub_9691E0((__int64)&v23, v4, 0, 0, 0);
      goto LABEL_11;
    case '&':
      v8 = &v21;
      sub_AB14C0((__int64)&v21, a3);
      if ( v22 <= 0x40 )
      {
        if ( v21 == (1LL << ((unsigned __int8)v22 - 1)) - 1 )
          goto LABEL_23;
      }
      else
      {
        v16 = v22 - 1;
        if ( !sub_986C60(&v21, v22 - 1) && v16 == (unsigned int)sub_C445E0(&v21) )
          goto LABEL_23;
      }
      sub_986680((__int64)&v27, v4);
LABEL_29:
      v17 = v22;
      v22 = 0;
      v24 = v17;
      v23 = v21;
      sub_C46A40(&v23, 1);
      v18 = v24;
      v24 = 0;
      v26 = v18;
      v25 = v23;
      sub_AADC30(a1, (__int64)&v25, &v27);
      sub_969240(&v25);
      sub_969240(&v23);
      sub_969240(&v27);
      goto LABEL_21;
    case '\'':
      v8 = &v27;
      sub_986680((__int64)&v27, v4);
      sub_AB14C0((__int64)&v25, a3);
LABEL_20:
      sub_9875E0(a1, &v25, &v27);
      sub_969240(&v25);
      goto LABEL_21;
    case '(':
      v8 = &v23;
      sub_AB13A0((__int64)&v23, a3);
      if ( (unsigned __int8)sub_986B30(&v23, a3, v9, v10, v11) )
        goto LABEL_23;
      v12 = v24;
      v24 = 0;
      v28 = v12;
      v27 = v23;
      sub_986680((__int64)&v25, v4);
LABEL_9:
      sub_AADC30(a1, (__int64)&v25, &v27);
      sub_969240(&v25);
      sub_969240(&v27);
      goto LABEL_21;
    case ')':
      sub_AB13A0((__int64)&v25, a3);
      sub_C46A40(&v25, 1);
      v13 = v26;
      v26 = 0;
      v28 = v13;
      v27 = v25;
      sub_986680((__int64)&v23, v4);
LABEL_11:
      sub_9875E0(a1, &v23, &v27);
      sub_969240(&v23);
      sub_969240(&v27);
      sub_969240(&v25);
      return a1;
    default:
      BUG();
  }
  return a1;
}
