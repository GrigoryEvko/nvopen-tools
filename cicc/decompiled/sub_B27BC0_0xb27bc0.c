// Function: sub_B27BC0
// Address: 0xb27bc0
//
__int64 __fastcall sub_B27BC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v5; // r15
  __int64 i; // r12
  __int64 v8; // r15
  __int64 *v9; // r14
  __int64 *v10; // r10
  __int64 v11; // r14
  __int64 v12; // r12
  __int64 v13; // r13
  __int64 v14; // r15
  __int64 *v15; // r14
  __int64 v17; // rdx
  __int64 v19; // [rsp+10h] [rbp-B0h]
  __int64 v20; // [rsp+28h] [rbp-98h]
  __int64 v22; // [rsp+40h] [rbp-80h]
  unsigned int v23; // [rsp+4Ch] [rbp-74h]
  unsigned int v24; // [rsp+4Ch] [rbp-74h]
  __int64 *v25[2]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v26; // [rsp+60h] [rbp-60h]
  __int64 *v27[2]; // [rsp+70h] [rbp-50h] BYREF
  __int64 v28; // [rsp+80h] [rbp-40h]

  v5 = a1;
  v19 = a3 & 1;
  v20 = (a3 - 1) / 2;
  if ( a2 >= v20 )
  {
    v11 = a2;
    v10 = (__int64 *)(a1 + 8 * a2);
    if ( (a3 & 1) != 0 )
      goto LABEL_13;
    goto LABEL_15;
  }
  for ( i = a2; ; i = v8 )
  {
    v8 = 2 * (i + 1);
    v9 = (__int64 *)(a1 + 16 * (i + 1));
    v22 = *(v9 - 1);
    sub_B1C5B0(v25, a5, *v9);
    v23 = *(_DWORD *)(v26 + 8);
    sub_B1C5B0(v27, a5, v22);
    if ( v23 < *(_DWORD *)(v28 + 8) )
    {
      --v8;
      v9 = (__int64 *)(a1 + 8 * v8);
    }
    *(_QWORD *)(a1 + 8 * i) = *v9;
    if ( v8 >= v20 )
      break;
  }
  v10 = v9;
  v11 = v8;
  v5 = a1;
  if ( !v19 )
  {
LABEL_15:
    if ( (a3 - 2) / 2 == v11 )
    {
      v17 = *(_QWORD *)(v5 + 8 * (2 * v11 + 2) - 8);
      v11 = 2 * v11 + 1;
      *v10 = v17;
      v10 = (__int64 *)(v5 + 8 * v11);
    }
  }
  if ( v11 > a2 )
  {
    v12 = (v11 - 1) / 2;
    v13 = v5;
    v14 = v11;
    while ( 1 )
    {
      v15 = (__int64 *)(v13 + 8 * v12);
      sub_B1C5B0(v25, a5, *v15);
      v24 = *(_DWORD *)(v26 + 8);
      sub_B1C5B0(v27, a5, a4);
      v10 = (__int64 *)(v13 + 8 * v14);
      if ( v24 >= *(_DWORD *)(v28 + 8) )
        break;
      v14 = v12;
      *v10 = *v15;
      if ( a2 >= v12 )
      {
        v10 = (__int64 *)(v13 + 8 * v12);
        break;
      }
      v12 = (v12 - 1) / 2;
    }
  }
LABEL_13:
  *v10 = a4;
  return a4;
}
