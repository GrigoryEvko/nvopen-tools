// Function: sub_D72B90
// Address: 0xd72b90
//
unsigned __int64 __fastcall sub_D72B90(_QWORD *a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 v12; // r12
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  _QWORD *v17; // rax
  _QWORD *v18; // r8
  _QWORD *v19; // rdi
  _QWORD v20[2]; // [rsp+0h] [rbp-40h] BYREF
  unsigned __int64 v21; // [rsp+10h] [rbp-30h]

  v4 = a2;
  v20[0] = 0;
  v20[1] = 0;
  v21 = a2;
  if ( a2 != 0 && a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)v20);
  if ( a1[94] )
  {
    v17 = (_QWORD *)a1[91];
    v18 = a1 + 90;
    if ( v17 )
    {
      v19 = a1 + 90;
      do
      {
        if ( v17[6] < v21 )
        {
          v17 = (_QWORD *)v17[3];
        }
        else
        {
          v19 = v17;
          v17 = (_QWORD *)v17[2];
        }
      }
      while ( v17 );
      if ( v18 != v19 && v21 >= v19[6] )
      {
LABEL_10:
        sub_D68D70(v20);
        return v4;
      }
    }
  }
  else
  {
    v6 = a1[63];
    v7 = v6 + 24LL * *((unsigned int *)a1 + 128);
    if ( v6 != v7 )
    {
      while ( *(_QWORD *)(v6 + 16) != v21 )
      {
        v6 += 24;
        if ( v7 == v6 )
          goto LABEL_14;
      }
      if ( v7 != v6 )
        goto LABEL_10;
    }
  }
LABEL_14:
  sub_D68D70(v20);
  v10 = *a3;
  v11 = *a3 + 24LL * *((unsigned int *)a3 + 2);
  if ( v11 == *a3 )
    return *(_QWORD *)(*a1 + 128LL);
  v12 = 0;
  do
  {
    v13 = *(_QWORD *)(v10 + 16);
    if ( v13 != v12 && a2 != v13 )
    {
      if ( v12 )
        return v4;
      v12 = *(_QWORD *)(v10 + 16);
    }
    v10 += 24;
  }
  while ( v10 != v11 );
  if ( !v12 )
    return *(_QWORD *)(*a1 + 128LL);
  if ( a2 )
  {
    sub_BD84D0(a2, v12);
    sub_D6E4B0(a1, a2, 0, v14, v15, v16);
  }
  return sub_D6D330((__int64)a1, v12, v13, v11, v9);
}
