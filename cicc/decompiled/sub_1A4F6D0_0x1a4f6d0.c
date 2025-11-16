// Function: sub_1A4F6D0
// Address: 0x1a4f6d0
//
__int64 __fastcall sub_1A4F6D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 i; // r15
  __int64 v6; // r12
  __int64 *v7; // rbx
  __int64 v8; // r14
  __int64 *v9; // r15
  __int64 v11; // rax
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h]
  __int64 v16; // [rsp+28h] [rbp-48h] BYREF
  __int64 v17[7]; // [rsp+38h] [rbp-38h] BYREF

  v16 = a5;
  v14 = (a3 - 1) / 2;
  v13 = a3 & 1;
  if ( a2 >= v14 )
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
    if ( sub_1A4F560(&v16, *v7, *(v7 - 1)) )
    {
      --v6;
      v7 = (__int64 *)(a1 + 8 * v6);
    }
    *(_QWORD *)(a1 + 8 * i) = *v7;
    if ( v6 >= v14 )
      break;
  }
  if ( !v13 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == v6 )
    {
      v11 = *(_QWORD *)(a1 + 8 * (2 * v6 + 2) - 8);
      v6 = 2 * v6 + 1;
      *v7 = v11;
      v7 = (__int64 *)(a1 + 8 * v6);
    }
  }
  v17[0] = v16;
  v8 = (v6 - 1) / 2;
  if ( v6 > a2 )
  {
    while ( 1 )
    {
      v9 = (__int64 *)(a1 + 8 * v8);
      v7 = (__int64 *)(a1 + 8 * v6);
      if ( !sub_1A4F560(v17, *v9, a4) )
        break;
      v6 = v8;
      *v7 = *v9;
      if ( a2 >= v8 )
      {
        v7 = (__int64 *)(a1 + 8 * v8);
        break;
      }
      v8 = (v8 - 1) / 2;
    }
  }
LABEL_13:
  *v7 = a4;
  return a4;
}
