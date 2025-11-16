// Function: sub_2648E10
// Address: 0x2648e10
//
char *__fastcall sub_2648E10(char *a1, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *v10; // rbx
  __int64 v11; // rdi
  __int64 v12; // rcx
  __int64 v13; // rax
  volatile signed __int32 *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  volatile signed __int32 *v18; // rdi
  __int64 v19; // r14
  __int64 v20; // r13
  _QWORD *v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // rax
  volatile signed __int32 *v24; // rdi
  __int64 v25; // rax
  __int64 v27; // r13
  _QWORD *v28; // r14
  __int64 v29; // rcx
  __int64 v30; // rax
  volatile signed __int32 *v31; // rdi
  unsigned int v32; // [rsp+Ch] [rbp-84h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  _QWORD v35[4]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v36[10]; // [rsp+40h] [rbp-50h] BYREF

  if ( a2 == a1 )
    goto LABEL_14;
  v10 = a1;
  while ( a4 != a3 )
  {
    v15 = *(_QWORD *)a3;
    v16 = *(_QWORD *)v10;
    if ( !*(_DWORD *)(*(_QWORD *)a3 + 40LL) )
      goto LABEL_11;
    if ( *(_DWORD *)(v16 + 40) )
    {
      v11 = *(unsigned __int8 *)(v15 + 16);
      v12 = *(unsigned __int8 *)(v16 + 16);
      if ( (_BYTE)v11 == (_BYTE)v12 )
      {
        sub_22B0690(v35, (__int64 *)(v15 + 24));
        v32 = *(_DWORD *)v35[2];
        sub_22B0690(v36, (__int64 *)(*(_QWORD *)v10 + 24LL));
        if ( v32 >= *(_DWORD *)v36[2] )
        {
          v16 = *(_QWORD *)v10;
          goto LABEL_11;
        }
        v15 = *(_QWORD *)a3;
        goto LABEL_6;
      }
      if ( *(_DWORD *)(a6 + 4 * v11) < *(_DWORD *)(a6 + 4 * v12) )
        goto LABEL_6;
LABEL_11:
      *(_QWORD *)v10 = 0;
      v17 = *((_QWORD *)v10 + 1);
      *((_QWORD *)v10 + 1) = 0;
      v18 = (volatile signed __int32 *)a5[1];
      *a5 = v16;
      a5[1] = v17;
      if ( v18 )
        sub_A191D0(v18);
      v10 += 16;
      a5 += 2;
      if ( a2 == v10 )
        goto LABEL_14;
    }
    else
    {
LABEL_6:
      *(_QWORD *)a3 = 0;
      v13 = *((_QWORD *)a3 + 1);
      *((_QWORD *)a3 + 1) = 0;
      v14 = (volatile signed __int32 *)a5[1];
      *a5 = v15;
      a5[1] = v13;
      if ( v14 )
        sub_A191D0(v14);
      a3 += 16;
      a5 += 2;
      if ( a2 == v10 )
        goto LABEL_14;
    }
  }
  v33 = a2 - v10;
  v27 = (a2 - v10) >> 4;
  if ( v33 <= 0 )
    return (char *)a5;
  v28 = a5;
  do
  {
    v29 = *(_QWORD *)v10;
    v30 = *((_QWORD *)v10 + 1);
    *(_QWORD *)v10 = 0;
    *((_QWORD *)v10 + 1) = 0;
    v31 = (volatile signed __int32 *)v28[1];
    *v28 = v29;
    v28[1] = v30;
    if ( v31 )
      sub_A191D0(v31);
    v10 += 16;
    v28 += 2;
    --v27;
  }
  while ( v27 );
  a5 = (_QWORD *)((char *)a5 + v33);
LABEL_14:
  v19 = a4 - a3;
  v20 = (a4 - a3) >> 4;
  if ( a4 - a3 <= 0 )
    return (char *)a5;
  v21 = a5;
  do
  {
    v22 = *(_QWORD *)a3;
    v23 = *((_QWORD *)a3 + 1);
    *(_QWORD *)a3 = 0;
    *((_QWORD *)a3 + 1) = 0;
    v24 = (volatile signed __int32 *)v21[1];
    *v21 = v22;
    v21[1] = v23;
    if ( v24 )
      sub_A191D0(v24);
    a3 += 16;
    v21 += 2;
    --v20;
  }
  while ( v20 );
  v25 = 16;
  if ( v19 > 0 )
    v25 = v19;
  return (char *)a5 + v25;
}
