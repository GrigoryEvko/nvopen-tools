// Function: sub_28EB420
// Address: 0x28eb420
//
__int64 __fastcall sub_28EB420(unsigned __int8 *a1)
{
  __int64 v1; // r15
  unsigned int v2; // r12d
  int v3; // edx
  __int64 v4; // rsi
  unsigned __int8 *v5; // rdx
  __int64 v6; // r13
  __int64 v7; // r12
  __int64 v8; // rdx
  unsigned __int8 *v9; // rax
  unsigned __int8 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 *v13; // r13
  __int64 v14; // rsi
  __int64 v16; // rsi
  unsigned __int8 *v17; // rsi
  __int64 v18[4]; // [rsp+0h] [rbp-60h] BYREF
  __int16 v19; // [rsp+20h] [rbp-40h]

  v1 = *((_QWORD *)a1 + 1);
  v2 = *a1 - 42;
  v3 = *(unsigned __int8 *)(v1 + 8);
  if ( (unsigned int)(v3 - 17) <= 1 )
    LOBYTE(v3) = *(_BYTE *)(**(_QWORD **)(v1 + 16) + 8LL);
  if ( (_BYTE)v3 == 12 )
    v4 = sub_AD62B0(*((_QWORD *)a1 + 1));
  else
    v4 = (__int64)sub_AD8DD0(*((_QWORD *)a1 + 1), -1.0);
  v19 = 257;
  if ( (a1[7] & 0x40) != 0 )
    v5 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v5 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v6 = v2 < 0x12 ? 0x20 : 0;
  v7 = sub_28E92A0(*(_QWORD *)&v5[v6], v4, (__int64)v18, (__int64)(a1 + 24), 0, 0, (__int64)a1);
  v8 = sub_AD6530(v1, v4);
  if ( (a1[7] & 0x40) != 0 )
    v9 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
  else
    v9 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
  v10 = &v9[v6];
  if ( *(_QWORD *)v10 )
  {
    v11 = *((_QWORD *)v10 + 1);
    **((_QWORD **)v10 + 2) = v11;
    if ( v11 )
      *(_QWORD *)(v11 + 16) = *((_QWORD *)v10 + 2);
  }
  *(_QWORD *)v10 = v8;
  if ( v8 )
  {
    v12 = *(_QWORD *)(v8 + 16);
    *((_QWORD *)v10 + 1) = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = v10 + 8;
    *((_QWORD *)v10 + 2) = v8 + 16;
    *(_QWORD *)(v8 + 16) = v10;
  }
  v13 = (__int64 *)(v7 + 48);
  sub_BD6B90((unsigned __int8 *)v7, a1);
  sub_BD84D0((__int64)a1, v7);
  v14 = *((_QWORD *)a1 + 6);
  v18[0] = v14;
  if ( !v14 )
  {
    if ( v13 == v18 )
      return v7;
    v16 = *(_QWORD *)(v7 + 48);
    if ( !v16 )
      return v7;
LABEL_24:
    sub_B91220(v7 + 48, v16);
    goto LABEL_25;
  }
  sub_B96E90((__int64)v18, v14, 1);
  if ( v13 == v18 )
  {
    if ( v18[0] )
      sub_B91220((__int64)v18, v18[0]);
    return v7;
  }
  v16 = *(_QWORD *)(v7 + 48);
  if ( v16 )
    goto LABEL_24;
LABEL_25:
  v17 = (unsigned __int8 *)v18[0];
  *(_QWORD *)(v7 + 48) = v18[0];
  if ( v17 )
    sub_B976B0((__int64)v18, v17, v7 + 48);
  return v7;
}
