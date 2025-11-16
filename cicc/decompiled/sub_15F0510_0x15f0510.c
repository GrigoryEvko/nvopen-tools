// Function: sub_15F0510
// Address: 0x15f0510
//
__int64 __fastcall sub_15F0510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r15
  __int64 v7; // r12
  unsigned __int64 v8; // r13
  __int64 v9; // rax
  __int64 *v10; // r8
  __int64 v11; // r13
  __int64 *v12; // r15
  __int64 v13; // rax
  __int64 v15; // rax
  __int64 v17; // [rsp+10h] [rbp-60h]
  __int64 v18; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-40h]
  __int64 *v21; // [rsp+38h] [rbp-38h]
  unsigned __int64 v22; // [rsp+38h] [rbp-38h]

  v18 = (a3 - 1) / 2;
  v17 = a3 & 1;
  if ( a2 >= v18 )
  {
    v7 = a2;
    v10 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v7 )
  {
    v7 = 2 * (i + 1);
    v21 = (__int64 *)(a1 + 16 * (i + 1));
    v20 = *(v21 - 1);
    v8 = *(_QWORD *)(sub_15EFCB0(a5, *v21) + 784);
    v9 = sub_15EFCB0(a5, v20);
    v10 = v21;
    if ( v8 > *(_QWORD *)(v9 + 784) )
    {
      --v7;
      v10 = (__int64 *)(a1 + 8 * v7);
    }
    *(_QWORD *)(a1 + 8 * i) = *v10;
    if ( v7 >= v18 )
      break;
  }
  if ( !v17 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v7 )
    {
      v15 = *(_QWORD *)(a1 + 8 * (2 * v7 + 2) - 8);
      v7 = 2 * v7 + 1;
      *v10 = v15;
      v10 = (__int64 *)(a1 + 8 * v7);
    }
  }
  v11 = (v7 - 1) / 2;
  if ( v7 > a2 )
  {
    while ( 1 )
    {
      v12 = (__int64 *)(a1 + 8 * v11);
      v22 = *(_QWORD *)(sub_15EFCB0(a5, *v12) + 784);
      v13 = sub_15EFCB0(a5, a4);
      v10 = (__int64 *)(a1 + 8 * v7);
      if ( v22 <= *(_QWORD *)(v13 + 784) )
        break;
      v7 = v11;
      *v10 = *v12;
      if ( a2 >= v11 )
      {
        v10 = (__int64 *)(a1 + 8 * v11);
        break;
      }
      v11 = (v11 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = a4;
  return a4;
}
