// Function: sub_B8D360
// Address: 0xb8d360
//
__int64 __fastcall sub_B8D360(_QWORD *a1, __int64 *a2, __int64 a3)
{
  __int64 v5; // r14
  _QWORD *v6; // rax
  unsigned __int64 v7; // rbx
  _OWORD *i; // rdx
  __int64 v9; // rbx
  __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // r13
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  _OWORD *v18; // rsi
  __int64 v19; // r12
  _OWORD *v22; // [rsp+10h] [rbp-60h] BYREF
  __int64 v23; // [rsp+18h] [rbp-58h]
  _OWORD v24[5]; // [rsp+20h] [rbp-50h] BYREF

  v5 = sub_BCB2E0(*a1);
  v6 = v24;
  v23 = 0x400000000LL;
  v22 = v24;
  v7 = 2 * a3;
  if ( v7 )
  {
    if ( v7 > 4 )
    {
      sub_C8D5F0(&v22, v24, v7, 8);
      v6 = (_QWORD *)v22 + (unsigned int)v23;
      for ( i = &v22[a3]; i != (_OWORD *)v6; ++v6 )
      {
LABEL_4:
        if ( v6 )
          *v6 = 0;
      }
    }
    else
    {
      i = &v24[a3];
      if ( i != v24 )
        goto LABEL_4;
    }
    LODWORD(v23) = v7;
  }
  v9 = 0;
  if ( a3 )
  {
    do
    {
      v10 = *a2;
      v11 = a2[1];
      v12 = v9++;
      a2 += 3;
      v13 = sub_B8C130(a1, v10, v11);
      *(_QWORD *)&v22[v12] = v13;
      v14 = sub_ACD640(v5, *(a2 - 1), 0);
      v17 = sub_B8C140((__int64)a1, v14, v15, v16);
      *((_QWORD *)&v22[v12] + 1) = v17;
    }
    while ( a3 != v9 );
  }
  v18 = v22;
  v19 = sub_B9C770(*a1, v22, (unsigned int)v23, 0, 1);
  if ( v22 != v24 )
    _libc_free(v22, v18);
  return v19;
}
