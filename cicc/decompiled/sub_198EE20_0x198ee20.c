// Function: sub_198EE20
// Address: 0x198ee20
//
__int64 __fastcall sub_198EE20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r12
  __int64 v6; // r15
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // r14
  __int64 v10; // r12
  __int64 v12; // rax
  __int64 v14; // [rsp+10h] [rbp-70h]
  __int64 v16; // [rsp+30h] [rbp-50h]
  __int64 v17; // [rsp+38h] [rbp-48h] BYREF
  __int64 v18[7]; // [rsp+48h] [rbp-38h] BYREF

  v17 = a5;
  v16 = (a3 - 1) / 2;
  v14 = a3 & 1;
  if ( a2 >= v16 )
  {
    v7 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    v6 = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = v6 )
  {
    v6 = 2 * (i + 1);
    v7 = (__int64 *)(a1 + 16 * (i + 1));
    v8 = *v7;
    if ( sub_198ECB0(&v17, *v7, *(v7 - 1)) )
    {
      --v6;
      v7 = (__int64 *)(a1 + 8 * v6);
      v8 = *v7;
    }
    *(_QWORD *)(a1 + 8 * i) = v8;
    if ( v6 >= v16 )
      break;
  }
  if ( !v14 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v6 )
    {
      v12 = *(_QWORD *)(a1 + 8 * (2 * v6 + 2) - 8);
      v6 = 2 * v6 + 1;
      *v7 = v12;
      v7 = (__int64 *)(a1 + 8 * v6);
    }
  }
  v18[0] = v17;
  v9 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v7 = (__int64 *)(a1 + 8 * v6);
      v10 = *(_QWORD *)(a1 + 8 * v9);
      if ( !sub_198ECB0(v18, v10, a4) )
        break;
      *v7 = v10;
      v6 = v9;
      if ( a2 >= v9 )
      {
        v7 = (__int64 *)(a1 + 8 * v9);
        break;
      }
      v9 = (v9 - 1) / 2;
    }
  }
LABEL_13:
  *v7 = a4;
  return a4;
}
