// Function: sub_2649060
// Address: 0x2649060
//
_QWORD *__fastcall sub_2649060(char *a1, char *a2, char *a3, char *a4, _QWORD *a5, __int64 a6)
{
  char *v6; // r15
  char *v9; // rbx
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // rcx
  volatile signed __int32 *v14; // rdi
  __int64 v15; // rsi
  __int64 v16; // rcx
  __int64 v17; // rsi
  volatile signed __int32 *v18; // rdi
  __int64 v19; // r12
  _QWORD *v20; // r14
  __int64 v21; // rsi
  __int64 v22; // rcx
  volatile signed __int32 *v23; // rdi
  __int64 v24; // r14
  __int64 v25; // r12
  _QWORD *v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rcx
  volatile signed __int32 *v29; // rdi
  __int64 v30; // rax
  unsigned int v32; // [rsp+Ch] [rbp-84h]
  __int64 v33; // [rsp+10h] [rbp-80h]
  _QWORD v35[4]; // [rsp+20h] [rbp-70h] BYREF
  _QWORD v36[10]; // [rsp+40h] [rbp-50h] BYREF

  v6 = a3;
  v9 = a1;
  if ( a3 != a4 && a1 != a2 )
  {
    do
    {
      v15 = *(_QWORD *)v6;
      v16 = *(_QWORD *)v9;
      if ( !*(_DWORD *)(*(_QWORD *)v6 + 40LL) )
        goto LABEL_12;
      if ( *(_DWORD *)(v16 + 40) )
      {
        v11 = *(unsigned __int8 *)(v15 + 16);
        v12 = *(unsigned __int8 *)(v16 + 16);
        if ( (_BYTE)v11 == (_BYTE)v12 )
        {
          sub_22B0690(v35, (__int64 *)(v15 + 24));
          v32 = *(_DWORD *)v35[2];
          sub_22B0690(v36, (__int64 *)(*(_QWORD *)v9 + 24LL));
          if ( v32 >= *(_DWORD *)v36[2] )
          {
            v16 = *(_QWORD *)v9;
LABEL_12:
            *(_QWORD *)v9 = 0;
            v17 = *((_QWORD *)v9 + 1);
            *((_QWORD *)v9 + 1) = 0;
            v18 = (volatile signed __int32 *)a5[1];
            *a5 = v16;
            a5[1] = v17;
            if ( v18 )
              sub_A191D0(v18);
            v9 += 16;
            a5 += 2;
            if ( v9 == a2 )
              break;
            continue;
          }
          v15 = *(_QWORD *)v6;
        }
        else if ( *(_DWORD *)(a6 + 4 * v11) >= *(_DWORD *)(a6 + 4 * v12) )
        {
          goto LABEL_12;
        }
      }
      *(_QWORD *)v6 = 0;
      v13 = *((_QWORD *)v6 + 1);
      *((_QWORD *)v6 + 1) = 0;
      v14 = (volatile signed __int32 *)a5[1];
      *a5 = v15;
      a5[1] = v13;
      if ( v14 )
        sub_A191D0(v14);
      v6 += 16;
      a5 += 2;
      if ( v9 == a2 )
        break;
    }
    while ( v6 != a4 );
  }
  v33 = a2 - v9;
  v19 = (a2 - v9) >> 4;
  if ( v33 > 0 )
  {
    v20 = a5;
    do
    {
      v21 = *(_QWORD *)v9;
      v22 = *((_QWORD *)v9 + 1);
      *(_QWORD *)v9 = 0;
      *((_QWORD *)v9 + 1) = 0;
      v23 = (volatile signed __int32 *)v20[1];
      *v20 = v21;
      v20[1] = v22;
      if ( v23 )
        sub_A191D0(v23);
      v9 += 16;
      v20 += 2;
      --v19;
    }
    while ( v19 );
    a5 = (_QWORD *)((char *)a5 + v33);
  }
  v24 = a4 - v6;
  v25 = (a4 - v6) >> 4;
  if ( a4 - v6 > 0 )
  {
    v26 = a5;
    do
    {
      v27 = *(_QWORD *)v6;
      v28 = *((_QWORD *)v6 + 1);
      *(_QWORD *)v6 = 0;
      *((_QWORD *)v6 + 1) = 0;
      v29 = (volatile signed __int32 *)v26[1];
      *v26 = v27;
      v26[1] = v28;
      if ( v29 )
        sub_A191D0(v29);
      v6 += 16;
      v26 += 2;
      --v25;
    }
    while ( v25 );
    v30 = 16;
    if ( v24 > 0 )
      v30 = v24;
    return (_QWORD *)((char *)a5 + v30);
  }
  return a5;
}
