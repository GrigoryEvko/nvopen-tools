// Function: sub_B44C30
// Address: 0xb44c30
//
__int64 __fastcall sub_B44C30(unsigned __int8 *a1, __int64 *a2)
{
  __int64 result; // rax
  __int64 v3; // rdx
  unsigned int v4; // ebx
  int v5; // edx
  int v6; // eax
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rbx
  __int64 v10; // r12
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r15
  int v15; // r15d
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  unsigned __int64 v19; // rax
  int v20; // edx
  _BYTE v21[32]; // [rsp+0h] [rbp-70h] BYREF
  __int64 v22; // [rsp+20h] [rbp-50h]

  sub_B9ADA0();
  result = (unsigned int)*a1 - 34;
  if ( (unsigned __int8)(*a1 - 34) > 0x33u )
    return result;
  v3 = 0x8000000000041LL;
  if ( !_bittest64(&v3, result) || !*((_QWORD *)a1 + 9) )
    return result;
  v4 = 0;
  sub_A753E0((__int64)v21);
  v5 = *a1;
  v6 = v5 - 29;
  if ( v5 != 40 )
    goto LABEL_5;
LABEL_22:
  v7 = 32LL * (unsigned int)sub_B491D0(a1);
  if ( (a1[7] & 0x80u) == 0 )
  {
LABEL_23:
    v18 = 0;
    goto LABEL_20;
  }
  while ( 1 )
  {
    v12 = sub_BD2BC0(a1);
    v14 = v12 + v13;
    if ( (a1[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v14 >> 4) )
LABEL_25:
        BUG();
      goto LABEL_23;
    }
    if ( !(unsigned int)((v14 - sub_BD2BC0(a1)) >> 4) )
      goto LABEL_23;
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_25;
    v15 = *(_DWORD *)(sub_BD2BC0(a1) + 8);
    if ( (a1[7] & 0x80u) == 0 )
      BUG();
    v16 = sub_BD2BC0(a1);
    v18 = 32LL * (unsigned int)(*(_DWORD *)(v16 + v17 - 4) - v15);
LABEL_20:
    if ( v4 >= (unsigned int)((32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF) - 32 - v7 - v18) >> 5) )
      break;
    ++v4;
    a2 = (__int64 *)sub_BD5C60(a1, a2);
    v19 = sub_A7A440((__int64 *)a1 + 9, a2, v4, (__int64)v21);
    v20 = *a1;
    *((_QWORD *)a1 + 9) = v19;
    v6 = v20 - 29;
    if ( v20 == 40 )
      goto LABEL_22;
LABEL_5:
    v7 = 0;
    if ( v6 != 56 )
    {
      if ( v6 != 5 )
        BUG();
      v7 = 64;
    }
    if ( (a1[7] & 0x80u) == 0 )
      goto LABEL_23;
  }
  v8 = sub_BD5C60(a1, a2);
  result = sub_A7A440((__int64 *)a1 + 9, (__int64 *)v8, 0, (__int64)v21);
  v9 = v22;
  for ( *((_QWORD *)a1 + 9) = result; v9; result = j_j___libc_free_0(v10, 88) )
  {
    v10 = v9;
    sub_B43850(*(_QWORD **)(v9 + 24), v8);
    v11 = *(_QWORD *)(v9 + 32);
    v9 = *(_QWORD *)(v9 + 16);
    if ( v11 != v10 + 56 )
      _libc_free(v11, v8);
    v8 = 88;
  }
  return result;
}
