// Function: sub_BD58A0
// Address: 0xbd58a0
//
__int64 __fastcall sub_BD58A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int8 *v4; // r12
  unsigned __int8 *v5; // r14
  unsigned __int8 v6; // al
  unsigned __int8 v7; // al
  unsigned int v8; // eax
  __int64 v10; // rbx
  unsigned int v11; // ecx
  unsigned int i; // r15d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 *v20; // [rsp+10h] [rbp-80h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-78h]
  __int64 *v22; // [rsp+20h] [rbp-70h] BYREF
  unsigned int v23; // [rsp+28h] [rbp-68h]
  __int64 v24; // [rsp+30h] [rbp-60h]
  __int64 v25; // [rsp+38h] [rbp-58h]
  __int64 v26; // [rsp+40h] [rbp-50h]
  __int64 v27; // [rsp+48h] [rbp-48h]
  __int64 v28; // [rsp+50h] [rbp-40h]
  __int64 v29; // [rsp+58h] [rbp-38h]

  v21 = sub_AE43F0(a3, *(_QWORD *)(a2 + 8));
  if ( v21 > 0x40 )
    sub_C43690(&v20, 0, 0);
  else
    v20 = 0;
  v23 = sub_AE43F0(a3, *(_QWORD *)(a1 + 8));
  if ( v23 > 0x40 )
    sub_C43690(&v22, 0, 0);
  else
    v22 = 0;
  v4 = sub_BD45C0((unsigned __int8 *)a2, a3, (__int64)&v20, 1, 0, 0, 0, 0);
  v5 = sub_BD45C0((unsigned __int8 *)a1, a3, (__int64)&v22, 1, 0, 0, 0, 0);
  if ( v4 == v5 )
  {
    v8 = v23;
    if ( v23 <= 0x40 )
    {
      v19 = 0;
      if ( v23 )
        v19 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23);
    }
    else
    {
      v19 = *v22;
    }
    if ( v21 > 0x40 )
    {
      v19 -= *v20;
    }
    else if ( v21 )
    {
      v19 -= (__int64)((_QWORD)v20 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
    }
    goto LABEL_49;
  }
  v6 = *v4;
  if ( *v4 > 0x1Cu )
  {
    if ( v6 != 63 )
      v4 = 0;
LABEL_9:
    v7 = *v5;
    if ( *v5 > 0x1Cu )
      goto LABEL_10;
LABEL_23:
    if ( v7 != 5 || *((_WORD *)v5 + 1) != 34 )
      goto LABEL_11;
    goto LABEL_25;
  }
  if ( v6 != 5 )
  {
    v4 = 0;
    goto LABEL_9;
  }
  if ( *((_WORD *)v4 + 1) != 34 )
    v4 = 0;
  v7 = *v5;
  if ( *v5 <= 0x1Cu )
    goto LABEL_23;
LABEL_10:
  if ( v7 == 63 )
  {
LABEL_25:
    if ( !v4 )
      goto LABEL_11;
    if ( *(_QWORD *)&v5[-32 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)] != *(_QWORD *)&v4[-32
                                                                                 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)] )
      goto LABEL_11;
    v10 = sub_BB5290((__int64)v4);
    if ( v10 != sub_BB5290((__int64)v5) )
      goto LABEL_11;
    v11 = *((_DWORD *)v4 + 1) & 0x7FFFFFF;
    if ( v11 == 1 )
    {
      i = 1;
    }
    else
    {
      for ( i = 1; i != v11; ++i )
      {
        if ( (*((_DWORD *)v5 + 1) & 0x7FFFFFF) == i )
          break;
        if ( *(_QWORD *)&v5[32 * (i - (unsigned __int64)(*((_DWORD *)v5 + 1) & 0x7FFFFFF))] != *(_QWORD *)&v4[32 * (i - (unsigned __int64)v11)] )
          break;
      }
    }
    v13 = sub_BD3110((__int64)v4, i, a3);
    v25 = v14;
    v24 = v13;
    v26 = sub_BD3110((__int64)v5, i, a3);
    v27 = v15;
    if ( !(_BYTE)v25 || !(_BYTE)v27 )
      goto LABEL_11;
    v8 = v23;
    if ( v23 > 0x40 )
    {
      v16 = *v22;
    }
    else
    {
      v16 = 0;
      if ( v23 )
        v16 = (__int64)((_QWORD)v22 << (64 - (unsigned __int8)v23)) >> (64 - (unsigned __int8)v23);
    }
    v17 = v16 + v26 - v24;
    if ( v21 > 0x40 )
    {
      v18 = *v20;
    }
    else
    {
      v18 = 0;
      if ( v21 )
        v18 = (__int64)((_QWORD)v20 << (64 - (unsigned __int8)v21)) >> (64 - (unsigned __int8)v21);
    }
    v19 = v17 - v18;
LABEL_49:
    v28 = v19;
    LOBYTE(v29) = 1;
    goto LABEL_12;
  }
LABEL_11:
  LOBYTE(v29) = 0;
  v8 = v23;
LABEL_12:
  if ( v8 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  return v28;
}
