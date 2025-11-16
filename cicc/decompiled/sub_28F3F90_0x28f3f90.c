// Function: sub_28F3F90
// Address: 0x28f3f90
//
__int64 __fastcall sub_28F3F90(unsigned __int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rcx
  unsigned __int8 *v4; // rax
  __int64 v5; // rcx
  unsigned __int8 *v6; // rsi
  __int64 *v7; // rdi
  __int64 v8; // r12
  __int64 v9; // rdx
  __int64 *v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // rax
  __int64 v15; // rcx
  __int64 v16; // rcx
  __int64 *v17; // r14
  __int64 v18; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 v22[4]; // [rsp+0h] [rbp-50h] BYREF
  __int16 v23; // [rsp+20h] [rbp-30h]

  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a1 - 8);
  else
    v3 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  v4 = sub_28F38B0(*(_QWORD *)(v3 + 32), a1, a2);
  v5 = a1 + 24;
  v23 = 257;
  v6 = v4;
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v7 = *(__int64 **)(a1 - 8);
  else
    v7 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  v8 = sub_28E9200(*v7, (__int64)v4, (__int64)v22, v5, 0, 0, a1);
  v9 = sub_AD6530(*(_QWORD *)(a1 + 8), (__int64)v6);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v10 = *(__int64 **)(a1 - 8);
  else
    v10 = (__int64 *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *v10 )
  {
    v6 = (unsigned __int8 *)v10[2];
    v11 = v10[1];
    *(_QWORD *)v6 = v11;
    if ( v11 )
    {
      v6 = (unsigned __int8 *)v10[2];
      *(_QWORD *)(v11 + 16) = v6;
    }
  }
  *v10 = v9;
  if ( v9 )
  {
    v12 = *(_QWORD *)(v9 + 16);
    v6 = (unsigned __int8 *)(v9 + 16);
    v10[1] = v12;
    if ( v12 )
      *(_QWORD *)(v12 + 16) = v10 + 1;
    v10[2] = (__int64)v6;
    *(_QWORD *)(v9 + 16) = v10;
  }
  v13 = sub_AD6530(*(_QWORD *)(a1 + 8), (__int64)v6);
  if ( (*(_BYTE *)(a1 + 7) & 0x40) != 0 )
    v14 = *(_QWORD *)(a1 - 8);
  else
    v14 = a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
  if ( *(_QWORD *)(v14 + 32) )
  {
    v15 = *(_QWORD *)(v14 + 40);
    **(_QWORD **)(v14 + 48) = v15;
    if ( v15 )
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v14 + 48);
  }
  *(_QWORD *)(v14 + 32) = v13;
  if ( v13 )
  {
    v16 = *(_QWORD *)(v13 + 16);
    *(_QWORD *)(v14 + 40) = v16;
    if ( v16 )
      *(_QWORD *)(v16 + 16) = v14 + 40;
    *(_QWORD *)(v14 + 48) = v13 + 16;
    *(_QWORD *)(v13 + 16) = v14 + 32;
  }
  v17 = (__int64 *)(v8 + 48);
  sub_BD6B90((unsigned __int8 *)v8, (unsigned __int8 *)a1);
  sub_BD84D0(a1, v8);
  v18 = *(_QWORD *)(a1 + 48);
  v22[0] = v18;
  if ( v18 )
  {
    sub_B96E90((__int64)v22, v18, 1);
    if ( v17 == v22 )
    {
      if ( v22[0] )
        sub_B91220((__int64)v22, v22[0]);
      return v8;
    }
    v20 = *(_QWORD *)(v8 + 48);
    if ( !v20 )
    {
LABEL_34:
      v21 = (unsigned __int8 *)v22[0];
      *(_QWORD *)(v8 + 48) = v22[0];
      if ( v21 )
      {
        sub_B976B0((__int64)v22, v21, v8 + 48);
        return v8;
      }
      return v8;
    }
LABEL_33:
    sub_B91220(v8 + 48, v20);
    goto LABEL_34;
  }
  if ( v17 == v22 )
    return v8;
  v20 = *(_QWORD *)(v8 + 48);
  if ( v20 )
    goto LABEL_33;
  return v8;
}
